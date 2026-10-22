"""Request/response models (A7). The web console's TypeScript types are generated from
the OpenAPI schema these produce, so field names here are the contract."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TxnType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
Action = Literal["APPROVE", "STEP_UP", "REVIEW", "BLOCK"]
SetLabel = Literal["legit", "fraud"]
LabelSource = Literal["analyst", "chargeback", "simulator"]
CaseStatus = Literal["open", "in_review", "resolved"]
MAX_BATCH = 1_000


class ScoreRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "txn_id": "T000123",
                    "step": 30,
                    "type": "TRANSFER",
                    "amount": 181000.0,
                    "name_orig": "C1305486145",
                    "name_dest": "C553264065",
                }
            ]
        },
    )

    txn_id: str = Field(min_length=1, max_length=64, examples=["T000123"])
    step: int = Field(ge=1, le=744, description="Hour index (PaySim step).")
    type: TxnType
    amount: float = Field(ge=0, le=1e10)
    name_orig: str = Field(pattern=r"^[CM]\d+$", max_length=32)
    name_dest: str = Field(pattern=r"^[CM]\d+$", max_length=32)
    oldbalance_orig: float | None = Field(
        default=None, ge=0, description="Accepted for compatibility; not used by the model (D2)."
    )


class ErrorBody(BaseModel):
    code: str
    message: str


class Reason(BaseModel):
    code: str
    text: str
    impact: float


class ScoreResponse(BaseModel):
    txn_id: str
    model_version: str
    p_fraud: float
    decision: Action
    decision_reason: str
    expected_cost: dict[str, float]
    prediction_set: list[SetLabel] | None = Field(
        default=None,
        description='Conformal set: ["legit"], ["fraud"], both ("model uncertain") or [] (empty).',
    )
    reasons: list[Reason] = []
    explanation_stable: bool | None = Field(
        default=None,
        description="False when the top reasons would likely change under a refit (5.2).",
    )
    latency_ms: float

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "txn_id": "T000123",
                    "model_version": "lgbm_g1-2026.10.21-a1b2c3",
                    "p_fraud": 0.412,
                    "prediction_set": ["legit", "fraud"],
                    "decision": "STEP_UP",
                    "decision_reason": "model_uncertain",
                    "expected_cost": {"APPROVE": 1830.2, "STEP_UP": 560.0, "REVIEW": 2188.0},
                    "reasons": [
                        {
                            "code": "DEST_HISTORY",
                            "text": "Receiving account has never received money before",
                            "impact": 0.87,
                        }
                    ],
                    "explanation_stable": True,
                    "latency_ms": 3.1,
                }
            ]
        }
    )


class ErrorResponse(BaseModel):
    error: ErrorBody


class Health(BaseModel):
    status: Literal["ok"]


class Ready(BaseModel):
    status: Literal["ready", "not_ready"]
    model_version: str | None = None
    detail: str | None = None
    checks: dict[str, bool] = {}


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[ScoreRequest] = Field(
        min_length=1, max_length=MAX_BATCH, description="Scored in the order given."
    )


class BatchItemError(BaseModel):
    txn_id: str
    error: ErrorBody


class BatchResponse(BaseModel):
    results: list[ScoreResponse | BatchItemError]
    scored: int
    failed: int


# --------------------------------------------------------------------- stored data


class TransactionOut(BaseModel):
    txn_id: str
    step: int
    type: str
    amount: float
    name_orig: str
    name_dest: str
    received_at: dt.datetime
    scenario: str | None = None


class LabelOut(BaseModel):
    txn_id: str
    is_fraud: bool
    source: LabelSource
    labelled_at_step: int


class StoredPrediction(BaseModel):
    model_version: str
    p_fraud: float
    prediction_set: list[SetLabel] | None
    decision: Action
    decision_reason: str
    expected_cost: dict[str, float]
    reasons: list[Reason]
    explanation_stable: bool | None
    latency_ms: float


class TransactionDetail(BaseModel):
    transaction: TransactionOut
    prediction: StoredPrediction | None
    features: dict[str, float | None] = Field(
        description="Engine output the model saw; null = no history (filled at scoring)."
    )
    label: LabelOut | None


class ExplainResponse(BaseModel):
    txn_id: str
    model_version: str
    contributions: dict[str, float] = Field(description="TreeSHAP per feature (log-odds).")
    bias: float = Field(description="Expected margin; bias + sum(contributions) = margin.")
    groups: dict[str, float] = Field(description="Contributions summed per reason code.")
    reasons: list[Reason]
    explanation_stable: bool


class GraphNode(BaseModel):
    id: str
    kind: Literal["customer", "merchant"]
    center: bool


class GraphEdge(BaseModel):
    txn_id: str
    source: str
    target: str
    step: int
    type: str
    amount: float
    is_fraud: bool | None = Field(description="Known label, null if unlabelled.")


class GraphResponse(BaseModel):
    txn_id: str
    hops: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool


class CaseOut(BaseModel):
    case_id: int
    txn_id: str
    priority: float = Field(description="Expected savings of a review over approving.")
    status: CaseStatus
    assignee: str | None
    verdict: Literal["fraud", "legit"] | None
    notes: str | None
    created_at: dt.datetime
    resolved_at: dt.datetime | None
    step: int
    type: str
    amount: float
    p_fraud: float | None
    decision: Action | None


class CasePage(BaseModel):
    items: list[CaseOut]
    total: int
    limit: int
    offset: int


class ResolveRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"verdict": "fraud", "assignee": "analyst-7"}]},
    )

    verdict: Literal["fraud", "legit"]
    assignee: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=2_000)


class ResolveResponse(BaseModel):
    case: CaseOut
    label: LabelOut
    label_created: bool


class LabelRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"txn_id": "T000123", "observed_fraud": True, "step": 54}]},
    )

    txn_id: str = Field(min_length=1, max_length=64)
    observed_fraud: bool
    step: int = Field(ge=1, le=744, description="Step at which the label became known.")
    source: LabelSource = "chargeback"


class LabelResponse(BaseModel):
    label: LabelOut
    created: bool = Field(description="False if the transaction was already labelled.")


class ModelPerformance(BaseModel):
    model_version: str
    role: Literal["champion", "challenger"]
    n_labelled: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None


class PerformanceResponse(BaseModel):
    window_steps: int
    since_step: int
    flagged: list[Action] = Field(description="Decisions counted as a fraud flag.")
    models: list[ModelPerformance]


class DriftWindowOut(BaseModel):
    window_id: int
    start_step: int
    end_step: int
    feature: str
    psi: float | None
    ks_stat: float | None
    ks_p: float | None
    alarm: bool


class AlarmOut(BaseModel):
    alarm_id: int
    kind: str
    detector: str
    feature: str | None
    step: int
    details: dict[str, Any]


class DriftResponse(BaseModel):
    window_steps: int
    since_step: int
    windows: list[DriftWindowOut]
    alarms: list[AlarmOut]


class ModelOut(BaseModel):
    version: str
    stage: Literal["champion", "challenger", "archived"]
    metrics: dict[str, Any]
    bundle_uri: str
    promoted_at: dt.datetime | None


class CurrentModel(BaseModel):
    version: str
    model: str
    created_at: str
    engine_version: str
    features: list[str]
    metrics: dict[str, Any]


class LabelBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[LabelRequest] = Field(min_length=1, max_length=MAX_BATCH)


class LabelBatchResponse(BaseModel):
    created: int
    already_labelled: int
    unknown: list[str] = Field(description="txn_ids with no stored transaction (skipped).")
    invalid: list[str] = Field(description="txn_ids labelled before their own step (skipped).")


class PromoteResponse(BaseModel):
    version: str
    previous: str


ScenarioId = Literal["D0", "D1", "D2", "D3", "D4", "D5", "D6"]


class SimStart(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"first_step": 356, "last_step": 408, "scenario": "D0"}]},
    )

    first_step: int = Field(default=356, ge=1, le=744)
    last_step: int = Field(default=408, ge=1, le=744)
    steps_per_tick: int = Field(default=1, ge=1, le=24, description="Steps replayed per tick.")
    scenario: ScenarioId = "D0"


class SimScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"name": "D3"}]})

    name: ScenarioId


class SimStatus(BaseModel):
    running: bool
    scenario: ScenarioId
    scenario_start: int = Field(description="Step from which the scenario's drift applies.")
    first_step: int
    last_step: int
    next_step: int
    steps_per_tick: int
    updated_at: dt.datetime
