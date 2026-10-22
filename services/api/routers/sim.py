"""Demo controls for the stream replayer and drift injection (admin only).

The replayer and the scenario library are Phase 6 work driven by the worker (7.4); the
routes exist now so the contract and its auth are fixed, and answer 501 until then.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Request
from slowapi import Limiter

from services.api.deps import AppContext, error_responses, limited, requires
from services.api.security import ApiError

PENDING = "replayer control arrives with the worker (task 7.4)"


def build(limiter: Limiter, ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/v1/sim", tags=["simulation"])
    admin = ctx.settings.rate_limit_admin
    errors = error_responses(401, 403, 429, 501)

    @router.post(
        "/start", dependencies=[requires("admin")], responses=errors, summary="Start the replayer"
    )
    @limited(limiter, admin)
    def start(request: Request) -> None:
        raise ApiError(501, "not_implemented", PENDING)

    @router.post(
        "/stop", dependencies=[requires("admin")], responses=errors, summary="Stop the replayer"
    )
    @limited(limiter, admin)
    def stop(request: Request) -> None:
        raise ApiError(501, "not_implemented", PENDING)

    @router.post(
        "/scenario",
        dependencies=[requires("admin")],
        responses=errors,
        summary="Inject a drift scenario (D0-D5)",
    )
    @limited(limiter, admin)
    def scenario(
        request: Request,
        name: Annotated[str, Body(embed=True, pattern=r"^D[0-5]$", examples=["D3"])],
    ) -> None:
        raise ApiError(501, "not_implemented", PENDING)

    return router
