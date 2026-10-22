"""`POST /v1/score` and `POST /v1/score/batch`."""

import structlog
from fastapi import APIRouter, BackgroundTasks, Request
from slowapi import Limiter

from fraudgraph.features.spec import Event
from fraudgraph.features.state import OutOfOrderError
from services.api.db import Scored
from services.api.deps import AppContext, context, error_responses, limited, requires
from services.api.schemas import (
    BatchItemError,
    BatchRequest,
    BatchResponse,
    ErrorBody,
    ScoreRequest,
    ScoreResponse,
)
from services.api.security import ApiError

decisions_log = structlog.stdlib.get_logger("fraudgraph.api")


def score_one(ctx: AppContext, request: ScoreRequest) -> tuple[ScoreResponse, Scored | None]:
    """Score and record one transaction. A repeated txn_id gets the first answer back and
    is not written again (the Redis marker also stops a retry on another worker from
    counting twice)."""
    with ctx.lock:
        scorer = ctx.require_scorer()  # read under the lock: a promotion swaps it
        cached = ctx.seen.get(request.txn_id)
        if cached is not None:
            return cached, None
        event = Event(
            txn_id=request.txn_id,
            step=request.step,
            type=request.type,
            amount=request.amount,
            name_orig=request.name_orig,
            name_dest=request.name_dest,
        )
        try:
            decision = scorer.score(event)
        except OutOfOrderError as exc:
            raise ApiError(409, "out_of_order", str(exc)) from exc
        fields = {k: v for k, v in decision.__dict__.items() if k != "features"}
        response = ScoreResponse(**fields)
        ctx.remember(response)
    ctx.metrics.decisions.labels(decision=response.decision).inc()
    decisions_log.info(
        "decision",
        txn_id=response.txn_id,
        model_version=response.model_version,
        decision=response.decision,
        decision_reason=response.decision_reason,
        latency_ms=response.latency_ms,
    )
    txn = request.model_dump(exclude_none=True)
    ctx.broadcaster.publish(
        {
            "kind": "decision",
            "txn_id": response.txn_id,
            "step": request.step,
            "type": request.type,
            "amount": request.amount,
            "decision": response.decision,
            "p_fraud": response.p_fraud,
            "model_version": response.model_version,
        }
    )
    return response, Scored(txn=txn, decision=decision, features=decision.features)


def build(limiter: Limiter, ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["scoring"])
    rules = ctx.settings

    @router.post(
        "/score",
        response_model=ScoreResponse,
        dependencies=[requires("write")],
        responses=error_responses(401, 403, 409, 413, 422, 429, 503),
        summary="Score one transaction",
    )
    @limited(limiter, rules.rate_limit_score)
    def score(request: Request, body: ScoreRequest, background: BackgroundTasks) -> ScoreResponse:
        """Decision, calibrated probability, conformal set, expected cost per action and
        reasons. The decision log is written after the response is sent."""
        response, scored = score_one(context(request), body)
        if scored is not None:
            background.add_task(context(request).write, [scored])
        return response

    @router.post(
        "/score/batch",
        response_model=BatchResponse,
        dependencies=[requires("write")],
        responses=error_responses(401, 403, 413, 422, 429, 503),
        summary="Score up to 1,000 transactions in order",
    )
    @limited(limiter, rules.rate_limit_batch)
    def score_batch(
        request: Request, body: BatchRequest, background: BackgroundTasks
    ) -> BatchResponse:
        """Each transaction is scored as if sent alone, in the given order. A failing item
        (e.g. out of order) gets an error entry and does not stop the rest."""
        ctx = context(request)
        ctx.require_scorer()
        results: list[ScoreResponse | BatchItemError] = []
        written: list[Scored] = []
        for item in body.transactions:
            try:
                response, scored = score_one(ctx, item)
            except ApiError as exc:
                results.append(
                    BatchItemError(
                        txn_id=item.txn_id, error=ErrorBody(code=exc.code, message=exc.message)
                    )
                )
                continue
            results.append(response)
            if scored is not None:
                written.append(scored)
        if written:
            background.add_task(ctx.write, written)
        failed = sum(isinstance(r, BatchItemError) for r in results)
        return BatchResponse(results=results, scored=len(results) - failed, failed=failed)

    return router
