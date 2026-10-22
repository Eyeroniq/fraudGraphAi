"""Stored transactions: detail, full explanation, ego-graph."""

import math
from typing import Annotated, Any

from fastapi import APIRouter, Path, Query, Request
from slowapi import Limiter

from fraudgraph.features.spec import Event
from services.api.db import Label, Prediction, Transaction
from services.api.deps import AppContext, context, error_responses, limited, requires
from services.api.schemas import (
    ExplainResponse,
    GraphEdge,
    GraphNode,
    GraphResponse,
    LabelOut,
    Reason,
    StoredPrediction,
    TransactionDetail,
    TransactionOut,
)
from services.api.security import ApiError

TxnId = Annotated[str, Path(min_length=1, max_length=64, examples=["T000123"])]
MAX_GRAPH_EDGES = 500


def label_out(label: Label) -> LabelOut:
    return LabelOut.model_validate(label, from_attributes=True)


def prediction_out(p: Prediction) -> StoredPrediction:
    sets: list[Any] | None = None
    if p.set_legit is not None and p.set_fraud is not None:
        sets = [c for c, inside in (("legit", p.set_legit), ("fraud", p.set_fraud)) if inside]
    return StoredPrediction(
        model_version=p.model_version,
        p_fraud=p.p_fraud,
        prediction_set=sets,
        decision=p.decision,  # type: ignore[arg-type]
        decision_reason=p.decision_reason,
        expected_cost=p.expected_cost,
        reasons=[Reason(**r) for r in p.reasons],
        explanation_stable=p.explanation_stable,
        latency_ms=p.latency_ms,
    )


def lookup(ctx: AppContext, txn_id: str) -> tuple[Transaction, Prediction | None, Label | None]:
    try:
        return ctx.log.transaction(txn_id)
    except KeyError:
        raise ApiError(404, "not_found", f"no transaction {txn_id}") from None


def build(limiter: Limiter, ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/v1/transactions", tags=["transactions"])
    read = ctx.settings.rate_limit_read
    errors = error_responses(401, 403, 404, 429)

    @router.get(
        "/{txn_id}",
        response_model=TransactionDetail,
        dependencies=[requires("read")],
        responses=errors,
        summary="Stored decision, features and label",
    )
    @limited(limiter, read)
    def transaction(request: Request, txn_id: TxnId) -> TransactionDetail:
        txn, prediction, label = lookup(context(request), txn_id)
        return TransactionDetail(
            transaction=TransactionOut.model_validate(txn, from_attributes=True),
            prediction=None if prediction is None else prediction_out(prediction),
            features={} if prediction is None else prediction.features,
            label=None if label is None else label_out(label),
        )

    @router.get(
        "/{txn_id}/explain",
        response_model=ExplainResponse,
        dependencies=[requires("read")],
        responses=error_responses(401, 403, 404, 409, 429, 503),
        summary="Full SHAP vector, grouped reasons and stability flag",
    )
    @limited(limiter, read)
    def explain(request: Request, txn_id: TxnId) -> ExplainResponse:
        """Recomputed from the stored features, so it explains exactly the stored score."""
        ctx = context(request)
        scorer = ctx.require_scorer()
        txn, prediction, _ = lookup(ctx, txn_id)
        if prediction is None or not prediction.features:
            raise ApiError(404, "not_found", f"{txn_id} has no model score to explain")
        if prediction.model_version != scorer.version:
            raise ApiError(
                409,
                "model_changed",
                f"scored by {prediction.model_version}, serving {scorer.version}",
            )
        feats = {k: math.nan if v is None else float(v) for k, v in prediction.features.items()}
        event = Event(txn.txn_id, txn.step, txn.type, txn.amount, txn.name_orig, txn.name_dest)
        try:
            explanation = scorer.explain_features(feats, event)
        except RuntimeError as exc:
            raise ApiError(503, "not_ready", str(exc)) from exc
        return ExplainResponse(
            txn_id=txn_id,
            model_version=scorer.version,
            contributions=explanation.contributions,
            bias=explanation.bias,
            groups=explanation.groups,
            reasons=[Reason(**r) for r in explanation.reasons],
            explanation_stable=explanation.explanation_stable,
        )

    @router.get(
        "/{txn_id}/graph",
        response_model=GraphResponse,
        dependencies=[requires("read")],
        responses=errors,
        summary="Ego-graph around the transaction's accounts",
    )
    @limited(limiter, read)
    def graph(
        request: Request,
        txn_id: TxnId,
        hops: Annotated[int, Query(ge=1, le=2)] = 2,
        max_edges: Annotated[int, Query(ge=1, le=MAX_GRAPH_EDGES)] = 200,
    ) -> GraphResponse:
        """Transactions up to the payment's own step within `hops` of its two accounts,
        with known fraud labels (what an analyst could have seen at decision time)."""
        ctx = context(request)
        txn, _, _ = lookup(ctx, txn_id)
        rows = ctx.log.ego_graph(txn, hops, max_edges)
        center = {txn.name_orig, txn.name_dest}
        accounts: dict[str, None] = {}
        edges = []
        for t, label in rows:
            accounts.setdefault(t.name_orig)
            accounts.setdefault(t.name_dest)
            edges.append(
                GraphEdge(
                    txn_id=t.txn_id,
                    source=t.name_orig,
                    target=t.name_dest,
                    step=t.step,
                    type=t.type,
                    amount=t.amount,
                    is_fraud=None if label is None else label.is_fraud,
                )
            )
        nodes = [
            GraphNode(
                id=a, kind="merchant" if a.startswith("M") else "customer", center=a in center
            )
            for a in accounts
        ]
        return GraphResponse(
            txn_id=txn_id, hops=hops, nodes=nodes, edges=edges, truncated=len(rows) >= max_edges
        )

    return router
