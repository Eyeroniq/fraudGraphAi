"""Model registry: `GET /v1/models`, `GET /v1/models/current`, promotion (7.5)."""

from typing import Annotated

from fastapi import APIRouter, Path, Request
from slowapi import Limiter

from services.api.deps import AppContext, context, error_responses, limited, requires
from services.api.schemas import CurrentModel, ModelOut, PromoteResponse


def build(limiter: Limiter, ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/v1/models", tags=["models"])
    read = ctx.settings.rate_limit_read

    @router.get(
        "",
        response_model=list[ModelOut],
        dependencies=[requires("read")],
        responses=error_responses(401, 403, 429),
        summary="Known model versions, newest promotion first",
    )
    @limited(limiter, read)
    def models(request: Request) -> list[ModelOut]:
        rows = context(request).log.models()
        return [ModelOut.model_validate(m, from_attributes=True) for m in rows]

    @router.get(
        "/current",
        response_model=CurrentModel,
        dependencies=[requires("read")],
        responses=error_responses(401, 403, 429, 503),
        summary="The champion being served, with its bundle metrics",
    )
    @limited(limiter, read)
    def current(request: Request) -> CurrentModel:
        manifest = context(request).require_scorer().manifest
        return CurrentModel(
            version=manifest["version"],
            model=manifest.get("model", ""),
            created_at=manifest.get("created_at", ""),
            engine_version=manifest.get("engine_version", ""),
            features=manifest.get("features", []),
            metrics=manifest.get("metrics", {}),
        )

    @router.post(
        "/{version}/promote",
        response_model=PromoteResponse,
        dependencies=[requires("admin")],
        responses=error_responses(401, 403, 404, 409, 429, 503),
        summary="Promote a challenger to champion (hot swap, no downtime)",
    )
    @limited(limiter, ctx.settings.rate_limit_admin)
    def promote(request: Request, version: Annotated[str, Path(max_length=128)]) -> PromoteResponse:
        """Load and verify the version's bundle, then swap it in between two requests.
        Feature state carries over; the previous champion is archived."""
        ctx = context(request)
        previous = ctx.require_scorer().version
        return PromoteResponse(version=ctx.promote(version).version, previous=previous)

    return router
