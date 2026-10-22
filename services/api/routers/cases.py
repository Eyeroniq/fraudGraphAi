"""Review queue: `GET /v1/cases`, `GET /v1/cases/{id}`, `POST /v1/cases/{id}/resolve`."""

from typing import Annotated, Literal

from fastapi import APIRouter, Path, Query, Request
from slowapi import Limiter

from services.api.db import Prediction, ReviewCase, Transaction
from services.api.deps import AppContext, context, error_responses, limited, requires
from services.api.routers.labels import record_label
from services.api.schemas import CaseOut, CasePage, CaseStatus, ResolveRequest, ResolveResponse
from services.api.security import ApiError

CaseId = Annotated[int, Path(ge=1)]


def case_out(case: ReviewCase, txn: Transaction, prediction: Prediction | None) -> CaseOut:
    return CaseOut(
        case_id=case.case_id,
        txn_id=case.txn_id,
        priority=case.priority,
        status=case.status,  # type: ignore[arg-type]
        assignee=case.assignee,
        verdict=case.verdict,  # type: ignore[arg-type]
        notes=case.notes,
        created_at=case.created_at,
        resolved_at=case.resolved_at,
        step=txn.step,
        type=txn.type,
        amount=txn.amount,
        p_fraud=None if prediction is None else prediction.p_fraud,
        decision=None if prediction is None else prediction.decision,  # type: ignore[arg-type]
    )


def load_case(ctx: AppContext, case_id: int) -> CaseOut:
    case = ctx.log.case(case_id)
    if case is None:
        raise ApiError(404, "not_found", f"no case {case_id}")
    txn, prediction, _ = ctx.log.transaction(case.txn_id)
    return case_out(case, txn, prediction)


def build(limiter: Limiter, ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/v1/cases", tags=["cases"])
    read = ctx.settings.rate_limit_read

    @router.get(
        "",
        response_model=CasePage,
        dependencies=[requires("read")],
        responses=error_responses(401, 403, 422, 429),
        summary="Review queue",
    )
    @limited(limiter, read)
    def cases(
        request: Request,
        status: CaseStatus | Literal["all"] = "open",
        assignee: Annotated[str | None, Query(max_length=64)] = None,
        sort: Literal["priority", "created"] = "priority",
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> CasePage:
        """Open cases, highest expected savings first, by default. Filter by `status`
        (`all` for every case) and `assignee`; paginate with `limit` / `offset`."""
        page = context(request).log.cases(
            status=None if status == "all" else status,
            assignee=assignee,
            sort=sort,
            limit=limit,
            offset=offset,
        )
        items = [case_out(*row) for row in page.items]
        return CasePage(items=items, total=page.total, limit=limit, offset=offset)

    @router.get(
        "/{case_id}",
        response_model=CaseOut,
        dependencies=[requires("read")],
        responses=error_responses(401, 403, 404, 429),
        summary="One review case",
    )
    @limited(limiter, read)
    def case(request: Request, case_id: CaseId) -> CaseOut:
        return load_case(context(request), case_id)

    @router.post(
        "/{case_id}/resolve",
        response_model=ResolveResponse,
        dependencies=[requires("write")],
        responses=error_responses(401, 403, 404, 409, 422, 429),
        summary="Analyst verdict; becomes a label",
    )
    @limited(limiter, ctx.settings.rate_limit_score)
    def resolve(request: Request, case_id: CaseId, body: ResolveRequest) -> ResolveResponse:
        """Closes the case and stores the verdict as an `analyst` label (unless the
        transaction already had a label, e.g. a chargeback)."""
        ctx = context(request)
        current = load_case(ctx, case_id)
        if current.status == "resolved":
            raise ApiError(409, "already_resolved", f"case {case_id} is already resolved")
        txn, _, _ = ctx.log.transaction(current.txn_id)
        step = max(txn.step, ctx.scorer.engine.store.latest_step or 0) if ctx.scorer else txn.step
        label, created = record_label(ctx, txn, body.verdict == "fraud", step, "analyst")
        ctx.log.resolve_case(case_id, body.verdict, body.assignee, body.notes)
        return ResolveResponse(case=load_case(ctx, case_id), label=label, label_created=created)

    return router
