// Typed client for the scoring API. Types mirror services/api/schemas.py; in 8.1 they
// are generated from the OpenAPI schema instead of written by hand.

export type TxnType = "CASH_IN" | "CASH_OUT" | "DEBIT" | "PAYMENT" | "TRANSFER";
export type Action = "APPROVE" | "STEP_UP" | "REVIEW" | "BLOCK";
export type SetLabel = "legit" | "fraud";

export interface ScoreRequest {
  txn_id: string;
  step: number;
  type: TxnType;
  amount: number;
  name_orig: string;
  name_dest: string;
}

export interface ScoreResponse {
  txn_id: string;
  model_version: string;
  p_fraud: number;
  decision: Action;
  decision_reason: string;
  expected_cost: Partial<Record<Action, number>>;
  prediction_set: SetLabel[] | null;
  reasons: { code: string; text: string; impact: number }[];
  latency_ms: number;
}

export interface ApiError {
  error: { code: string; message: string };
}

// Same origin behind Caddy ("/api"); `npm run dev` talks to uvicorn directly.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export async function score(request: ScoreRequest): Promise<ScoreResponse> {
  const response = await fetch(`${API_BASE}/v1/score`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
  const body = (await response.json()) as ScoreResponse | ApiError;
  if (!response.ok) {
    const message = "error" in body ? body.error.message : `HTTP ${response.status}`;
    throw new Error(message);
  }
  return body as ScoreResponse;
}
