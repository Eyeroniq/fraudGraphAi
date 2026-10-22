"""`POST /v1/labels`: late labels (chargebacks, the stream replayer)."""

from fastapi import APIRouter, Request, Response
from slowapi import Limiter

from services.api.db import Transaction
from services.api.deps import AppContext, context, error_responses, limited, requires
from services.api.schemas import LabelOut, LabelRequest, LabelResponse
from services.api.security import ApiError


def record_label(
    ctx: AppContext, txn: Transaction, is_fraud: bool, step: int, source: str
) -> tuple[LabelOut, bool]:
    """Store a label (first one wins) and feed confirmed fraud to the feature state.

    The label is recorded at the later of `step` and the newest step already scored, so
    it only affects features from the next step on (D4) and never trips the
    out-of-order guard.
    """
    if step < txn.step:
        raise ApiError(422, "invalid_request", f"step {step} is before the transaction's step")
    created = ctx.log.add_label(txn.txn_id, is_fraud, source, step)
    if created and is_fraud and ctx.scorer is not None:
        with ctx.lock:
            engine = ctx.scorer.engine
            known_at = max(step, engine.store.latest_step or step)
            engine.apply_label(txn.name_dest, True, known_at, txn_id=txn.txn_id)
    _, _, label = ctx.log.transaction(txn.txn_id)
    assert label is not None
    return LabelOut.model_validate(label, from_attributes=True), created


def build(limiter: Limiter, ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["labels"])

    @router.post(
        "/labels",
        response_model=LabelResponse,
        status_code=201,
        dependencies=[requires("write")],
        responses={
            200: {"description": "Already labelled (idempotent)", "model": LabelResponse},
            **error_responses(401, 403, 404, 422, 429),
        },
        summary="Post a late label",
    )
    @limited(limiter, ctx.settings.rate_limit_score)
    def post_label(request: Request, body: LabelRequest, response: Response) -> LabelResponse:
        """`{txn_id, observed_fraud, step}`: the replayer posts these `L` steps after the
        transaction. Posting the same txn_id again changes nothing (200)."""
        ctx = context(request)
        try:
            txn, _, _ = ctx.log.transaction(body.txn_id)
        except KeyError:
            raise ApiError(404, "unknown_txn", f"no transaction {body.txn_id}") from None
        label, created = record_label(ctx, txn, body.observed_fraud, body.step, body.source)
        if not created:
            response.status_code = 200
        return LabelResponse(label=label, created=created)

    return router
