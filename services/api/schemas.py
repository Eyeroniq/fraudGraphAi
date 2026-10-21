"""Request/response models (A7). The web console's TypeScript types are generated from
the OpenAPI schema these produce, so field names here are the contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TxnType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
Action = Literal["APPROVE", "REVIEW", "BLOCK"]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    txn_id: str = Field(min_length=1, max_length=64, examples=["T000123"])
    step: int = Field(ge=1, le=744, description="Hour index (PaySim step).")
    type: TxnType
    amount: float = Field(ge=0, le=1e10)
    name_orig: str = Field(pattern=r"^[CM]\d+$", max_length=32)
    name_dest: str = Field(pattern=r"^[CM]\d+$", max_length=32)
    oldbalance_orig: float | None = Field(
        default=None, ge=0, description="Accepted for compatibility; not used by the model (D2)."
    )


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
    reasons: list[Reason] = []
    latency_ms: float


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class Health(BaseModel):
    status: Literal["ok"]


class Ready(BaseModel):
    status: Literal["ready", "not_ready"]
    model_version: str | None = None
    detail: str | None = None
