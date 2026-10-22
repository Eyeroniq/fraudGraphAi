import { describe, expect, it } from "vitest";
import type { Costs } from "@/lib/api";
import { APPROVE, bestAction, BLOCK, costRow, expectedCostRule, logGrid, REVIEW, STEP_UP, thresholdCurve } from "@/lib/cost";
import { age, count, money, pct, prob, setLabel } from "@/lib/format";
import { addDecision, backoff, emptySnapshot, type LiveDecision } from "@/lib/live";

const COSTS: Costs = { c_review: 2000, c_friction: 20000, catch_rate: 0.9, c_block: 0, step_up_enabled: true, c_step_up: 200, step_up_catch_rate: 0.7 };

describe("cost model (matches fraudgraph/decision/costs.py)", () => {
  it("builds the A6 cost table", () => {
    const row = costRow(100_000, COSTS);
    expect(row.legit).toEqual([0, 2000, 20000, 200]);
    [100_000, 12_000, 0, 30_200].forEach((v, i) => expect(row.fraud[i]).toBeCloseTo(v, 6));
  });

  it("picks the cheapest expected action", () => {
    expect(bestAction(0, 100_000, COSTS)).toBe(APPROVE);
    expect(bestAction(0.99, 100_000, COSTS)).toBe(BLOCK);
    // p = 0.01 on 1M: approve 10k, review 2.1k, step-up 3.2k, block 19.8k -> review
    expect(bestAction(0.01, 1_000_000, COSTS)).toBe(REVIEW);
    // p = 0.01 on 50k: approve 500, step-up 348.5, review 2.0k -> step-up
    expect(bestAction(0.01, 50_000, COSTS)).toBe(STEP_UP);
    expect(bestAction(0.01, 50_000, { ...COSTS, step_up_enabled: false })).toBe(APPROVE);
  });

  it("weights sample rows when costing a threshold", () => {
    const s = { p: [0.9, 0.001, 0.2], amount: [1000, 1000, 1000], y: [1, 0, 0], w: [1, 100, 10] };
    const [point] = thresholdCurve(s, COSTS, [0.1]);
    expect(point.tp).toBe(1);
    expect(point.fp).toBe(10);
    expect(point.tn).toBe(100);
    expect(point.cost).toBe(1 * 2100 + 10 * 2000);
    expect(point.reviewShare).toBeCloseTo(11 / 111);
    // the per-payment rule never costs more than the best threshold on the same rows
    expect(expectedCostRule(s, COSTS)).toBeLessThanOrEqual(point.cost);
  });

  it("spaces thresholds on a log scale", () => {
    const g = logGrid(1e-4, 1e-1, 4);
    expect(g[0]).toBeCloseTo(1e-4);
    expect(g[3]).toBeCloseTo(1e-1);
    expect(g[1] / g[0]).toBeCloseTo(g[2] / g[1]);
  });
});

describe("formatting", () => {
  it("abbreviates large numbers only", () => {
    expect(money(1234)).toBe("1,234");
    expect(money(2_500_000)).toBe("2.5M");
    expect(count(null)).toBe("—");
  });
  it("shows small shares with more digits", () => {
    expect(pct(0.123)).toBe("12.3%");
    expect(pct(0.0017)).toBe("0.17%");
  });
  it("writes tiny probabilities in exponent form", () => {
    expect(prob(0.4123)).toBe("0.412");
    expect(prob(0.00004)).toBe("4.0e-5");
  });
  it("names conformal sets", () => {
    expect(setLabel(["legit", "fraud"])).toBe("uncertain");
    expect(setLabel(["fraud"])).toBe("fraud");
    expect(setLabel([])).toBe("no fit");
    expect(setLabel(null)).toBeNull();
  });
  it("formats ages", () => {
    const now = Date.parse("2026-10-22T12:00:00Z");
    expect(age("2026-10-22T11:59:30Z", now)).toBe("30s");
    expect(age("2026-10-22T09:55:00Z", now)).toBe("2h 05m");
  });
});

describe("live feed", () => {
  const decision = (i: number, at: number, d: LiveDecision["decision"] = "APPROVE"): LiveDecision => ({
    kind: "decision",
    txn_id: `T${i}`,
    step: 1,
    type: "TRANSFER",
    amount: 10,
    decision: d,
    p_fraud: 0.1,
    model_version: "m",
    at,
  });

  it("keeps the newest first, counts and 5-second buckets", () => {
    let s = emptySnapshot();
    s = addDecision(s, decision(1, 1_000));
    s = addDecision(s, decision(2, 2_000, "REVIEW"));
    s = addDecision(s, decision(3, 6_000));
    expect(s.recent.map((d) => d.txn_id)).toEqual(["T3", "T2", "T1"]);
    expect(s.counts).toEqual({ APPROVE: 2, STEP_UP: 0, REVIEW: 1, BLOCK: 0 });
    expect(s.buckets.map((b) => b.t)).toEqual([0, 5_000]);
    expect(s.buckets[0].counts.REVIEW).toBe(1);
  });

  it("caps the recent list at 200", () => {
    let s = emptySnapshot();
    for (let i = 0; i < 250; i++) s = addDecision(s, decision(i, i));
    expect(s.recent).toHaveLength(200);
    expect(s.total).toBe(250);
  });

  it("backs off exponentially to 10 s with jitter", () => {
    const mid = () => 0.5;
    expect([0, 1, 2, 3, 4, 5, 10].map((n) => backoff(n, mid))).toEqual([500, 1000, 2000, 4000, 8000, 10000, 10000]);
    expect(backoff(0, () => 0)).toBe(400);
    expect(backoff(0, () => 1)).toBe(600);
  });
});
