"""Demo controls for the stream replayer and drift injection.

The API only records what should happen (`sim_control`); the worker (7.4) replays the
events, applies the scenario from `scenario_start` on (D0-D6, `configs/drift_scenarios.yaml`)
and posts the transactions and their delayed labels back to this API.
"""

from fastapi import APIRouter, Request
from slowapi import Limiter

from services.api.deps import AppContext, context, error_responses, limited, requires
from services.api.schemas import SimScenario, SimStart, SimStatus
from services.api.security import ApiError


def build(limiter: Limiter, ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/v1/sim", tags=["simulation"])
    admin = ctx.settings.rate_limit_admin
    errors = error_responses(401, 403, 422, 429)

    def status(request: Request) -> SimStatus:
        row = context(request).log.sim() or context(request).log.set_sim()
        return SimStatus.model_validate(row, from_attributes=True)

    @router.get(
        "/status",
        response_model=SimStatus,
        dependencies=[requires("read")],
        responses=error_responses(401, 403, 429),
        summary="Replayer state",
    )
    @limited(limiter, ctx.settings.rate_limit_read)
    def get_status(request: Request) -> SimStatus:
        return status(request)

    @router.post(
        "/start",
        response_model=SimStatus,
        dependencies=[requires("admin")],
        responses=errors,
        summary="Start (or restart) the replayer over a step range",
    )
    @limited(limiter, admin)
    def start(request: Request, body: SimStart) -> SimStatus:
        if body.last_step < body.first_step:
            raise ApiError(422, "invalid_request", "last_step is before first_step")
        context(request).log.set_sim(
            running=True,
            scenario=body.scenario,
            scenario_start=body.first_step,
            first_step=body.first_step,
            last_step=body.last_step,
            next_step=body.first_step,
            steps_per_tick=body.steps_per_tick,
        )
        return status(request)

    @router.post(
        "/stop",
        response_model=SimStatus,
        dependencies=[requires("admin")],
        responses=errors,
        summary="Pause the replayer (start resumes from the first step)",
    )
    @limited(limiter, admin)
    def stop(request: Request) -> SimStatus:
        context(request).log.set_sim(running=False)
        return status(request)

    @router.post(
        "/scenario",
        response_model=SimStatus,
        dependencies=[requires("admin")],
        responses=errors,
        summary="Inject a drift scenario (D0-D6) from the next replayed step",
    )
    @limited(limiter, admin)
    def scenario(request: Request, body: SimScenario) -> SimStatus:
        log = context(request).log
        current = log.sim() or log.set_sim()
        log.set_sim(scenario=body.name, scenario_start=current.next_step)
        return status(request)

    return router
