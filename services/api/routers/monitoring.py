"""Monitoring: drift windows and alarms, rolling performance, stored reports.

Drift windows, alarms and reports are produced by the worker (7.4); until it runs these
routes return empty lists / 404. Performance is computed here from labelled decisions.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import FileResponse
from slowapi import Limiter

from services.api.db import FLAGGED
from services.api.deps import AppContext, context, error_responses, limited, requires
from services.api.schemas import (
    AlarmOut,
    DriftResponse,
    DriftWindowOut,
    ModelPerformance,
    PerformanceResponse,
)
from services.api.security import ApiError

WINDOW = r"^[1-9][0-9]{0,3}[hdw]$"
UNIT_STEPS = {"h": 1, "d": 24, "w": 168}  # 1 step = 1 hour


def window_steps(text: str) -> int:
    return int(text[:-1]) * UNIT_STEPS[text[-1]]


def since(ctx: AppContext, steps: int) -> int:
    latest = ctx.log.latest_step() or 0
    return max(latest - steps + 1, 0)


def ratio(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 6)


def build(limiter: Limiter, ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/v1/monitoring", tags=["monitoring"])
    read = ctx.settings.rate_limit_read
    errors = error_responses(401, 403, 422, 429)
    window_query = Query(pattern=WINDOW, description="Steps back from the newest: 24h, 7d, 1w")

    @router.get(
        "/drift",
        response_model=DriftResponse,
        dependencies=[requires("read")],
        responses=errors,
        summary="PSI/KS per feature and alarms in a recent window",
    )
    @limited(limiter, read)
    def drift(request: Request, window: Annotated[str, window_query] = "24h") -> DriftResponse:
        ctx = context(request)
        steps = window_steps(window)
        start = since(ctx, steps)
        windows, alarms = ctx.log.drift(start)
        return DriftResponse(
            window_steps=steps,
            since_step=start,
            windows=[DriftWindowOut.model_validate(w, from_attributes=True) for w in windows],
            alarms=[AlarmOut.model_validate(a, from_attributes=True) for a in alarms],
        )

    @router.get(
        "/performance",
        response_model=PerformanceResponse,
        dependencies=[requires("read")],
        responses=errors,
        summary="Rolling precision/recall from labelled transactions",
    )
    @limited(limiter, read)
    def performance(
        request: Request, window: Annotated[str, window_query] = "7d"
    ) -> PerformanceResponse:
        """A decision other than APPROVE counts as a fraud flag. Champion and any shadow
        challengers (7.5) are compared on the same labelled transactions."""
        ctx = context(request)
        steps = window_steps(window)
        start = since(ctx, steps)
        champion = ctx.scorer.version if ctx.scorer else None
        models = [
            ModelPerformance(
                model_version=version,
                role="champion" if version == champion else "challenger",
                n_labelled=sum(c.values()),
                precision=ratio(c["tp"], c["tp"] + c["fp"]),
                recall=ratio(c["tp"], c["tp"] + c["fn"]),
                **c,
            )
            for version, c in sorted(ctx.log.performance(start).items())
        ]
        return PerformanceResponse(
            window_steps=steps,
            since_step=start,
            flagged=list(FLAGGED),  # type: ignore[arg-type]
            models=models,
        )

    @router.get(
        "/reports/{report_id}",
        dependencies=[requires("read")],
        response_class=FileResponse,
        responses={
            200: {"content": {"text/html": {}, "application/json": {}}},
            **error_responses(401, 403, 404, 429),
        },
        summary="A stored Evidently report",
    )
    @limited(limiter, read)
    def report(
        request: Request,
        report_id: Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")],
        format: Literal["html", "json"] = "html",
    ) -> FileResponse:
        """Served from `reports/evidently/<id>.<format>` (written by the worker)."""
        path = context(request).reports_dir / "evidently" / f"{report_id}.{format}"
        if not path.is_file():
            raise ApiError(404, "not_found", f"no {format} report {report_id}")
        media = "text/html" if format == "html" else "application/json"
        return FileResponse(path, media_type=media)

    return router
