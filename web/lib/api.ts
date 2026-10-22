// Typed client for the FraudGraph API. Request/response types are generated from the
// API's OpenAPI schema (`npm run gen:api` -> lib/schema.d.ts); only the console report
// (a file the API serves as-is) is described here by hand.

import type { components } from "./schema";

type S = components["schemas"];
export type ScoreRequest = S["ScoreRequest"];
export type ScoreResponse = S["ScoreResponse"];
export type Action = ScoreResponse["decision"];
export type TxnType = ScoreRequest["type"];
export type CaseOut = S["CaseOut"];
export type CasePage = S["CasePage"];
export type ResolveResponse = S["ResolveResponse"];
export type TransactionDetail = S["TransactionDetail"];
export type ExplainResponse = S["ExplainResponse"];
export type GraphResponse = S["GraphResponse"];
export type DriftResponse = S["DriftResponse"];
export type PerformanceResponse = S["PerformanceResponse"];
export type ModelOut = S["ModelOut"];
export type CurrentModel = S["CurrentModel"];
export type SimStatus = S["SimStatus"];
export type PromoteResponse = S["PromoteResponse"];
export type Scenario = S["SimScenario"]["name"];

export interface Metric {
  value: number;
  ci95: [number, number];
}

export interface Costs {
  c_review: number;
  c_friction: number;
  catch_rate: number;
  c_block: number;
  step_up_enabled: boolean;
  c_step_up: number;
  step_up_catch_rate: number;
}

export interface PolicyResult {
  cost: { total: number; ci95: [number, number] };
  action_share: Record<Action, number>;
}

export interface DriftScenario {
  id: string;
  kind: string;
  mode: string;
  start_step: number;
  detection: { input: number | null; prediction: number | null; performance: number | null };
  policies: Record<
    string,
    { ap_post: number; cost_post: number; rolling_ap: (number | null)[]; time_to_recover_steps: number | null }
  >;
  windows: { start: number; max_psi: number; score_psi: number }[];
  alarm_steps: number[];
}

/** `reports/console/lab.json` (`fgx console-report`), served by GET /v1/models/lab. */
export interface LabReport {
  generated_at: string;
  champion: string;
  models: Record<string, Record<string, Record<string, Metric>>>;
  pr_curves: Record<string, { ap: number; prevalence: number; recall: number[]; precision: number[] }>;
  sample: { p: number[]; amount: number[]; y: number[]; w: number[]; rows: number; fraud: number };
  policy: {
    costs: Costs;
    chosen: { alpha: number; score: string; mondrian: boolean };
    policies_test: Record<string, PolicyResult>;
    coverage_grid: {
      score: string;
      mondrian: boolean;
      alpha: number;
      test: { coverage: { overall: number; legit: number; fraud: number } };
    }[];
    review_load_vs_alpha: { alpha: number; review_share: number; reviews_per_day_mean: number; test_cost: number }[];
    done_when: { target: number; test_fraud_coverage: number };
  };
  explain: {
    studied: { flagged_in_test: number; sampled: number; quality_rows: number };
    global_mean_abs_group_shap: Record<string, number>;
    flagged_mean_group_shap: Record<string, number>;
    seed_stability: { median_jaccard: number; mean_jaccard: number; "share_jaccard_ge_0.8": number; median_tau: number };
    stability_threshold: { threshold: number; balanced_accuracy: number; share_flagged_unstable: number };
    perturbation: Record<string, number>;
    background: Record<string, number>;
    quality: Record<string, Record<string, { mean: number; ci95: [number, number]; n: number }>>;
    examples: { txn_id: string; action: Action; p_fraud: number; reasons: { code: string; text: string; impact: number }[] }[];
  };
  drift: {
    thresholds: Record<string, number>;
    scenarios: DriftScenario[];
    late_period: { start: number; fraud_rate: number; max_psi: number; score_psi: number; static_ap: number | null }[];
  };
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

// Same origin behind Caddy ("/api"); `npm run dev` talks to uvicorn directly.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";
export const MOCK = process.env.NEXT_PUBLIC_MOCK === "1";

type Method = "GET" | "POST";

async function http<T>(method: Method, path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const error = data?.error ?? { code: "http_error", message: `HTTP ${response.status}` };
    throw new ApiError(response.status, error.code, error.message);
  }
  return data as T;
}

// The mock backend (dev without the API, and the e2e tests) is only fetched when
// NEXT_PUBLIC_MOCK=1 at build time; production bundles never load it.
let mockModule: Promise<typeof import("./mock")> | null = null;
export function mockBackend(): Promise<typeof import("./mock")> {
  // A constant condition at build time: webpack drops the import (and the chunk) when
  // NEXT_PUBLIC_MOCK is not "1".
  return process.env.NEXT_PUBLIC_MOCK === "1"
    ? (mockModule ??= import("./mock"))
    : Promise.reject(new Error("the mock backend exists only in mock builds"));
}

export async function request<T>(method: Method, path: string, body?: unknown): Promise<T> {
  if (MOCK) return (await mockBackend()).handle(method, path, body) as Promise<T>;
  return http<T>(method, path, body);
}

const qs = (params: Record<string, string | number | undefined>) =>
  new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== "")
      .map(([k, v]) => [k, String(v)]),
  ).toString();

export const api = {
  score: (body: ScoreRequest) => request<ScoreResponse>("POST", "/v1/score", body),
  cases: (p: { status?: string; sort?: string; limit?: number; offset?: number }) =>
    request<CasePage>("GET", `/v1/cases?${qs(p)}`),
  case: (id: number) => request<CaseOut>("GET", `/v1/cases/${id}`),
  resolve: (id: number, verdict: "fraud" | "legit", notes?: string) =>
    request<ResolveResponse>("POST", `/v1/cases/${id}/resolve`, { verdict, notes }),
  transaction: (id: string) => request<TransactionDetail>("GET", `/v1/transactions/${encodeURIComponent(id)}`),
  explain: (id: string) => request<ExplainResponse>("GET", `/v1/transactions/${encodeURIComponent(id)}/explain`),
  graph: (id: string, hops = 2) =>
    request<GraphResponse>("GET", `/v1/transactions/${encodeURIComponent(id)}/graph?hops=${hops}`),
  drift: (window = "1w") => request<DriftResponse>("GET", `/v1/monitoring/drift?window=${window}`),
  performance: (window = "1w") => request<PerformanceResponse>("GET", `/v1/monitoring/performance?window=${window}`),
  models: () => request<ModelOut[]>("GET", "/v1/models"),
  currentModel: () => request<CurrentModel>("GET", "/v1/models/current"),
  lab: () => request<LabReport>("GET", "/v1/models/lab"),
  promote: (version: string) => request<PromoteResponse>("POST", `/v1/models/${encodeURIComponent(version)}/promote`),
  simStatus: () => request<SimStatus>("GET", "/v1/sim/status"),
  simStart: (body: { first_step: number; last_step: number; steps_per_tick: number; scenario: Scenario }) =>
    request<SimStatus>("POST", "/v1/sim/start", body),
  simStop: () => request<SimStatus>("POST", "/v1/sim/stop"),
  simScenario: (name: Scenario) => request<SimStatus>("POST", "/v1/sim/scenario", { name }),
};

/** WebSocket URL of the live decision stream (same origin through Caddy). */
export function liveUrl(): string {
  if (API_BASE.startsWith("http")) return `${API_BASE.replace(/^http/, "ws")}/ws/live`;
  const { protocol, host } = window.location;
  return `${protocol === "https:" ? "wss" : "ws"}://${host}${API_BASE}/ws/live`;
}
