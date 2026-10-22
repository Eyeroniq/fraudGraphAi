"""Model registry: `GET /v1/models`, `GET /v1/models/current`, promotion (7.5)."""

from typing import Annotated

from fastapi import APIRouter, Path, Request
from slowapi import Limiter

from services.api.deps import AppContext, context, error_responses, limited, requires
from services.api.schemas import CurrentModel, ModelOut
from services.api.security import ApiError


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
        dependencies=[requires("admin")],
        responses=error_responses(401, 403, 429, 501),
        summary="Promote a challenger to champion (task 7.5)",
    )
    @limited(limiter, ctx.settings.rate_limit_admin)
    def promote(request: Request, version: Annotated[str, Path(max_length=128)]) -> None:
        """Hot-swapping the bundle arrives with champion/challenger shadow scoring (7.5)."""
        raise ApiError(501, "not_implemented", "model promotion arrives with task 7.5")

    return router
