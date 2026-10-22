// In-browser stand-in for the API (NEXT_PUBLIC_MOCK=1): UI work without the backend and
// the e2e tests. Same routes and shapes as services/api; deterministic (seeded) data.
// Never part of a production bundle: lib/api.ts only imports it in mock builds.

import type {
  Action,
  CaseOut,
  CurrentModel,
  DriftResponse,
  ExplainResponse,
  GraphResponse,
  LabReport,
  ModelOut,
  PerformanceResponse,
  ScoreRequest,
  ScoreResponse,
  SimStatus,
  TransactionDetail,
} from "./api";
import { ApiError } from "./api";
import { bestAction } from "./cost";
import type { LiveDecision } from "./live";

// ------------------------------------------------------------------ randomness

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(7);
const pick = <T,>(xs: readonly T[]) => xs[Math.floor(rand() * xs.length)];
const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const lognormal = (median: number, sigma: number) => median * Math.exp(sigma * normal());

// ------------------------------------------------------------------ constants

const CHAMPION = "lgbm_g1-2026.10.21-9fe27bf";
const CHALLENGER = "lgbm_g1-r384-2026.10.22-9fe27bf";
const COSTS = {
  c_review: 2000,
  c_friction: 20000,
  catch_rate: 0.9,
  c_block: 0,
  step_up_enabled: true,
  c_step_up: 200,
  step_up_catch_rate: 0.7,
};
const ACTION_BY_INDEX: Action[] = ["APPROVE", "REVIEW", "BLOCK", "STEP_UP"];
export const FEATURES = [
  "t_log_amount",
  "t_hour",
  "t_is_transfer",
  "o_amount_z",
  "o_steps_since_last",
  "g_dest_age",
  "g_dest_in_n_1d",
  "g_dest_in_n_1w",
  "g_dest_senders_1d",
  "g_dest_senders_1w",
  "g_dest_in_share_1d",
  "g_dest_log_in_sum",
  "g_pair_n",
];
const GROUP_OF: Record<string, string> = {
  t_log_amount: "AMOUNT",
  t_hour: "TIME_OF_DAY",
  t_is_transfer: "PAYMENT_TYPE",
  o_amount_z: "SENDER_HISTORY",
  o_steps_since_last: "SENDER_HISTORY",
  g_pair_n: "PAIR_HISTORY",
};
const groupOf = (f: string) => GROUP_OF[f] ?? "DEST_HISTORY";
const REASON_TEXT: Record<string, string> = {
  DEST_HISTORY: "Receiving account has not received any payment in the last 7 days",
  AMOUNT: "Amount is in a range where fraud is more common",
  TIME_OF_DAY: "Time of day raises the risk score for this payment",
  PAYMENT_TYPE: "TRANSFER payments carry more fraud",
  SENDER_HISTORY: "Unusual amount for this sender",
  PAIR_HISTORY: "First payment between these two accounts",
};
const SCENARIOS = ["D0", "D1", "D2", "D3", "D4", "D5", "D6"] as const;

// ------------------------------------------------------------------ state

interface Row {
  txn: TransactionDetail["transaction"];
  pred: NonNullable<TransactionDetail["prediction"]>;
  features: Record<string, number | null>;
  label: TransactionDetail["label"];
  isFraud: boolean;
  shadow: Action;
}

const rows = new Map<string, Row>();
const cases: CaseOut[] = [];
const windows: DriftResponse["windows"] = [];
const alarms: DriftResponse["alarms"] = [];
const listeners = new Set<(d: LiveDecision) => void>();
let models: ModelOut[] = [
  {
    version: CHAMPION,
    stage: "champion",
    metrics: { eval: { test: { ap: { value: 0.467, ci95: [0.29, 0.63] } } } },
    bundle_uri: CHAMPION,
    promoted_at: "2026-10-21T18:02:11Z",
  },
  {
    version: CHALLENGER,
    stage: "challenger",
    metrics: { retrained_at_step: 384, calib_ap: 0.52, champion_calib_ap: 0.47 },
    bundle_uri: CHALLENGER,
    promoted_at: null,
  },
  {
    version: "lgbm-2026.10.21-48749db",
    stage: "archived",
    metrics: { eval: { test: { ap: { value: 0.357, ci95: [0.16, 0.6] } } } },
    bundle_uri: "lgbm-2026.10.21-48749db",
    promoted_at: "2026-10-21T09:40:00Z",
  },
];
let step = 300;
let txnCounter = 1;
let caseCounter = 1;
let alarmCounter = 1;
let windowCounter = 1;
let sim: SimStatus = {
  running: false,
  scenario: "D0",
  scenario_start: 356,
  first_step: 356,
  last_step: 408,
  next_step: 356,
  steps_per_tick: 1,
  updated_at: new Date().toISOString(),
};
let simTimer: ReturnType<typeof setInterval> | null = null;
let fraudRate = 0.03; // inflated so the demo shows cases quickly
let seeding = true; // seeded history gets older timestamps than live traffic

// ------------------------------------------------------------------ scoring

function features(amount: number, type: string, fraud: boolean): Record<string, number | null> {
  const f: Record<string, number | null> = {};
  for (const name of FEATURES) f[name] = Math.round(rand() * 1000) / 100;
  f.t_log_amount = Math.log10(amount);
  f.t_hour = step % 24;
  f.t_is_transfer = type === "TRANSFER" ? 1 : 0;
  f.g_dest_age = fraud ? null : Math.round(rand() * 160);
  f.g_dest_in_n_1w = fraud ? 0 : Math.round(rand() * 30);
  f.o_amount_z = null;
  return f;
}

function decide(req: ScoreRequest, fraud: boolean): Row {
  const modelled = req.type === "TRANSFER" || req.type === "CASH_OUT";
  const p = modelled ? (fraud ? 0.35 + 0.6 * rand() : Math.min(0.2, rand() ** 6)) : 0;
  let action: Action = modelled ? ACTION_BY_INDEX[bestAction(p, req.amount, COSTS)] : "APPROVE";
  let reason = modelled ? "expected_cost" : "type_not_modelled";
  const uncertain = modelled && p > 0.01 && p < 0.3;
  if (uncertain && action === "APPROVE") {
    action = "STEP_UP";
    reason = "model_uncertain";
  }
  if (modelled && p > 0.3 && action === "BLOCK" && rand() < 0.5) {
    action = "REVIEW";
    reason = "conformal_fraud";
  }
  const feats = modelled ? features(req.amount, req.type, fraud) : {};
  const groups = Object.keys(REASON_TEXT);
  const reasons =
    action === "APPROVE"
      ? []
      : groups
          .slice(0, 3 + Math.floor(rand() * 2))
          .map((code) => ({ code, text: REASON_TEXT[code], impact: Math.round((0.2 + rand() * 1.4) * 100) / 100 }))
          .sort((a, b) => b.impact - a.impact)
          .slice(0, 3);
  const expected = Object.fromEntries(
    (["APPROVE", "REVIEW", "BLOCK", "STEP_UP"] as Action[]).map((a, i) => {
      const legit = [0, COSTS.c_review, COSTS.c_friction, COSTS.c_step_up][i];
      const fr = [req.amount, COSTS.c_review + 0.1 * req.amount, 0, COSTS.c_step_up + 0.3 * req.amount][i];
      return [a, Math.round(((1 - p) * legit + p * fr) * 100) / 100];
    }),
  );
  const pred = {
    model_version: models.find((m) => m.stage === "champion")?.version ?? CHAMPION,
    p_fraud: p,
    prediction_set: modelled ? (uncertain ? ["legit", "fraud"] : p > 0.3 ? ["fraud"] : ["legit"]) : null,
    decision: action,
    decision_reason: reason,
    expected_cost: expected,
    reasons,
    explanation_stable: action === "APPROVE" ? null : rand() > 0.4,
    latency_ms: Math.round((1.5 + rand() * 6) * 10) / 10,
  } as Row["pred"];
  return {
    txn: {
      txn_id: req.txn_id,
      step: req.step,
      type: req.type,
      amount: req.amount,
      name_orig: req.name_orig,
      name_dest: req.name_dest,
      received_at: new Date().toISOString(),
      scenario: sim.running ? sim.scenario : null,
    },
    pred,
    features: feats,
    label: null,
    isFraud: fraud,
    shadow: action === "APPROVE" && fraud && rand() < 0.5 ? "REVIEW" : action,
  };
}

function record(row: Row): void {
  rows.set(row.txn.txn_id, row);
  if (row.pred.decision === "REVIEW") {
    const costs = row.pred.expected_cost as Record<string, number>;
    cases.push({
      case_id: caseCounter++,
      txn_id: row.txn.txn_id,
      priority: Math.round((costs.APPROVE - costs.REVIEW) * 100) / 100,
      status: "open",
      assignee: null,
      verdict: null,
      notes: null,
      created_at: new Date(Date.now() - (seeding ? rand() * 3 * 3600_000 : 0)).toISOString(),
      resolved_at: null,
      step: row.txn.step,
      type: row.txn.type,
      amount: row.txn.amount,
      decision: row.pred.decision,
      p_fraud: row.pred.p_fraud,
    });
  }
  const d: LiveDecision = {
    kind: "decision",
    txn_id: row.txn.txn_id,
    step: row.txn.step,
    type: row.txn.type,
    amount: row.txn.amount,
    decision: row.pred.decision as Action,
    p_fraud: row.pred.p_fraud,
    model_version: row.pred.model_version,
    at: Date.now(),
  };
  for (const l of listeners) l(d);
}

function randomTxn(fraud: boolean): ScoreRequest {
  const type = fraud ? pick(["TRANSFER", "CASH_OUT"] as const) : pick(["PAYMENT", "PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"] as const);
  const amount = Math.round(lognormal(fraud ? 400_000 : 60_000, 1.1) * 100) / 100;
  return {
    txn_id: `M${String(txnCounter++).padStart(7, "0")}`,
    step,
    type,
    amount,
    name_orig: `C${1000 + Math.floor(rand() * 300)}`,
    name_dest: fraud && rand() < 0.6 ? `C9${Math.floor(rand() * 40)}` : type === "PAYMENT" ? `M${100 + Math.floor(rand() * 60)}` : `C${2000 + Math.floor(rand() * 400)}`,
  };
}

function generate(n: number) {
  for (let i = 0; i < n; i++) {
    const fraud = rand() < fraudRate;
    record(decide(randomTxn(fraud), fraud));
  }
}

function labelDue() {
  // Labels arrive six steps late in the demo; one in ten frauds is never reported.
  for (const row of rows.values()) {
    if (row.label || row.txn.step > step - 6) continue;
    if (row.isFraud && rand() < 0.1) continue;
    row.label = { txn_id: row.txn.txn_id, is_fraud: row.isFraud, source: "simulator", labelled_at_step: step };
  }
}

function driftWindow(end: number) {
  const start = end - 23;
  const hot = ["D3", "D5", "D6"].includes(sim.scenario) && start >= sim.scenario_start;
  for (const feature of FEATURES) {
    const base = feature.startsWith("g_dest") ? 0.05 + rand() * 0.18 : rand() * 0.08;
    const psi = hot && feature.startsWith("g_dest") ? base + 0.25 + rand() * 0.6 : base;
    const alarm = psi > 0.32;
    windows.push({ window_id: windowCounter++, start_step: start, end_step: end, feature, psi, ks_stat: psi / 3, ks_p: alarm ? 1e-6 : 0.3, alarm });
    if (alarm) alarms.push({ alarm_id: alarmCounter++, kind: "input", detector: "ks", feature, step: end, details: { psi } });
  }
  if (hot && rand() < 0.7) {
    alarms.push({ alarm_id: alarmCounter++, kind: "performance", detector: "page_hinkley", feature: null, step: end, details: {} });
  }
}

function tick() {
  if (sim.next_step > sim.last_step) {
    stopSim();
    return;
  }
  step = sim.next_step;
  if (sim.scenario === "D3" || sim.scenario === "D6") fraudRate = step >= sim.scenario_start ? 0.12 : 0.03;
  generate(5 + Math.floor(rand() * 6));
  labelDue();
  if ((step - 300) % 24 === 23) driftWindow(step);
  sim = { ...sim, next_step: step + 1, updated_at: new Date().toISOString() };
}

function stopSim() {
  if (simTimer) clearInterval(simTimer);
  simTimer = null;
  sim = { ...sim, running: false, updated_at: new Date().toISOString() };
}

// Seed: two days of history so every page has something to show.
(function seed() {
  for (let s = 300; s < 348; s++) {
    step = s;
    generate(8);
    if ((s - 300) % 24 === 23) driftWindow(s);
  }
  labelDue();
  step = 348;
  seeding = false;
})();

// ------------------------------------------------------------------ routes

const notFound = (what: string) => new ApiError(404, "not_found", `no ${what}`);

function casesPage(q: URLSearchParams) {
  const status = q.get("status");
  const sort = q.get("sort") ?? "priority";
  const limit = Number(q.get("limit") ?? 50);
  const offset = Number(q.get("offset") ?? 0);
  let items = cases.filter((c) => !status || c.status === status);
  items = [...items].sort((a, b) =>
    sort === "priority" ? b.priority - a.priority : Date.parse(b.created_at) - Date.parse(a.created_at),
  );
  return { items: items.slice(offset, offset + limit), total: items.length, limit, offset };
}

function explain(id: string): ExplainResponse {
  const row = rows.get(id);
  if (!row) throw notFound(`transaction ${id}`);
  const contributions: Record<string, number> = {};
  for (const f of FEATURES) contributions[f] = Math.round((normal() * 0.35 + (row.isFraud ? 0.12 : -0.08)) * 1000) / 1000;
  const groups: Record<string, number> = {};
  for (const [f, v] of Object.entries(contributions)) groups[groupOf(f)] = (groups[groupOf(f)] ?? 0) + v;
  return {
    txn_id: id,
    model_version: row.pred.model_version,
    bias: -6.2,
    contributions,
    groups,
    reasons: row.pred.reasons,
    explanation_stable: row.pred.explanation_stable ?? true,
  };
}

function graph(id: string): GraphResponse {
  const row = rows.get(id);
  if (!row) throw notFound(`transaction ${id}`);
  const center = [row.txn.name_orig, row.txn.name_dest];
  const edges: GraphResponse["edges"] = [];
  for (const r of rows.values()) {
    if (edges.length > 28) break;
    if (center.includes(r.txn.name_orig) || center.includes(r.txn.name_dest)) {
      edges.push({ txn_id: r.txn.txn_id, source: r.txn.name_orig, target: r.txn.name_dest, step: r.txn.step, type: r.txn.type, amount: r.txn.amount, is_fraud: r.label ? r.label.is_fraud : null });
    }
  }
  // A second hop of synthetic senders into the receiving account (mule-like fan-in).
  const fanIn = row.isFraud ? 9 : 3;
  for (let i = 0; i < fanIn; i++) {
    edges.push({ txn_id: `G${id}${i}`, source: `C7${i}${id.slice(-2)}`, target: row.txn.name_dest, step: row.txn.step - Math.floor(rand() * 20), type: "TRANSFER", amount: Math.round(lognormal(30_000, 0.8)), is_fraud: row.isFraud && i % 3 === 0 ? true : null });
  }
  const ids = new Set(edges.flatMap((e) => [e.source, e.target]));
  return {
    txn_id: id,
    hops: 2,
    truncated: false,
    nodes: [...ids].map((n) => ({ id: n, kind: n.startsWith("M") ? "merchant" : "customer", center: center.includes(n) })),
    edges,
  };
}

function performance(): PerformanceResponse {
  const counts = (pick: (r: Row) => Action) => {
    const c = { tp: 0, fp: 0, fn: 0, tn: 0 };
    for (const r of rows.values()) {
      if (!r.label || r.pred.decision_reason === "type_not_modelled") continue;
      const flagged = pick(r) !== "APPROVE";
      const key = r.label.is_fraud ? (flagged ? "tp" : "fn") : flagged ? "fp" : "tn";
      c[key]++;
    }
    return c;
  };
  const champion = models.find((m) => m.stage === "champion")?.version ?? CHAMPION;
  const challenger = models.find((m) => m.stage === "challenger")?.version;
  const make = (version: string, role: "champion" | "challenger", c: ReturnType<typeof counts>) => ({
    model_version: version,
    role,
    n_labelled: c.tp + c.fp + c.fn + c.tn,
    ...c,
    precision: c.tp + c.fp ? c.tp / (c.tp + c.fp) : null,
    recall: c.tp + c.fn ? c.tp / (c.tp + c.fn) : null,
  });
  const out = [make(champion, "champion", counts((r) => r.pred.decision as Action))];
  if (challenger) out.push(make(challenger, "challenger", counts((r) => r.shadow)));
  return { window_steps: 168, since_step: Math.max(0, step - 167), flagged: ["STEP_UP", "REVIEW", "BLOCK"], models: out };
}

function current(): CurrentModel {
  const champion = models.find((m) => m.stage === "champion")!;
  return {
    version: champion.version,
    model: champion.version.split("-")[0],
    created_at: champion.promoted_at ?? "",
    engine_version: "1.3.0",
    features: FEATURES,
    metrics: champion.metrics,
  };
}

export async function handle(method: string, path: string, body?: unknown): Promise<unknown> {
  await new Promise((r) => setTimeout(r, 60 + rand() * 120)); // feel like a network
  const url = new URL(path, "http://mock");
  const p = url.pathname;
  let m: RegExpMatchArray | null;
  if (method === "POST" && p === "/v1/score") {
    const req = body as ScoreRequest;
    const fraud = req.amount > 500_000 && rand() < 0.5;
    const row = decide(req, fraud);
    record(row);
    return { ...row.pred, txn_id: req.txn_id } satisfies ScoreResponse;
  }
  if (p === "/v1/cases") return casesPage(url.searchParams);
  if ((m = p.match(/^\/v1\/cases\/(\d+)$/))) {
    const c = cases.find((x) => x.case_id === Number(m![1]));
    if (!c) throw notFound("case");
    return c;
  }
  if ((m = p.match(/^\/v1\/cases\/(\d+)\/resolve$/))) {
    const i = cases.findIndex((x) => x.case_id === Number(m![1]));
    if (i < 0) throw notFound("case");
    const verdict = (body as { verdict: "fraud" | "legit" }).verdict;
    cases[i] = { ...cases[i], status: "resolved", verdict, resolved_at: new Date().toISOString() };
    const row = rows.get(cases[i].txn_id)!;
    const created = row.label === null;
    if (created) row.label = { txn_id: row.txn.txn_id, is_fraud: verdict === "fraud", source: "analyst", labelled_at_step: step };
    return { case: cases[i], label: row.label, label_created: created };
  }
  if ((m = p.match(/^\/v1\/transactions\/([^/]+)$/))) {
    const row = rows.get(decodeURIComponent(m[1]));
    if (!row) throw notFound("transaction");
    return { transaction: row.txn, prediction: row.pred, label: row.label, features: row.features } satisfies TransactionDetail;
  }
  if ((m = p.match(/^\/v1\/transactions\/([^/]+)\/explain$/))) return explain(decodeURIComponent(m[1]));
  if ((m = p.match(/^\/v1\/transactions\/([^/]+)\/graph$/))) return graph(decodeURIComponent(m[1]));
  if (p === "/v1/monitoring/drift") {
    return { window_steps: 168, since_step: Math.max(0, step - 167), windows, alarms } satisfies DriftResponse;
  }
  if (p === "/v1/monitoring/performance") return performance();
  if (p === "/v1/models") return models;
  if (p === "/v1/models/current") return current();
  if (p === "/v1/models/lab") return (await import("./mock-lab")).lab() satisfies LabReport;
  if ((m = p.match(/^\/v1\/models\/([^/]+)\/promote$/))) {
    const version = decodeURIComponent(m[1]);
    if (!models.some((x) => x.version === version)) throw new ApiError(404, "unknown_model", `no model version ${version}`);
    const previous = current().version;
    models = models.map((x) =>
      x.version === version
        ? { ...x, stage: "champion", promoted_at: new Date().toISOString() }
        : x.stage === "champion"
          ? { ...x, stage: "archived" }
          : x,
    );
    return { version, previous };
  }
  if (p === "/v1/sim/status") return sim;
  if (p === "/v1/sim/start") {
    const b = body as { first_step: number; last_step: number; steps_per_tick: number; scenario: (typeof SCENARIOS)[number] };
    if (b.last_step < b.first_step) throw new ApiError(422, "invalid_request", "last_step is before first_step");
    sim = { ...sim, ...b, running: true, scenario_start: b.first_step, next_step: Math.max(b.first_step, step + 1), updated_at: new Date().toISOString() };
    if (simTimer) clearInterval(simTimer);
    simTimer = setInterval(tick, 700);
    return sim;
  }
  if (p === "/v1/sim/stop") {
    stopSim();
    return sim;
  }
  if (p === "/v1/sim/scenario") {
    const name = (body as { name: string }).name;
    if (!SCENARIOS.includes(name as (typeof SCENARIOS)[number])) throw new ApiError(422, "invalid_request", "unknown scenario");
    sim = { ...sim, scenario: name as SimStatus["scenario"], scenario_start: sim.next_step, updated_at: new Date().toISOString() };
    return sim;
  }
  throw new ApiError(404, "not_found", `${method} ${p}`);
}

// A gentle background trickle while the replayer is stopped, so the demo looks alive.
let trickle: ReturnType<typeof setInterval> | null = null;

export function subscribeLive(listener: (d: LiveDecision) => void): () => void {
  listeners.add(listener);
  trickle ??= setInterval(() => {
    if (!sim.running) generate(1 + Math.floor(rand() * 2));
  }, 900);
  return () => listeners.delete(listener);
}
