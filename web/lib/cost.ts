// The A6 cost model (fraudgraph/decision/costs.py), in the browser: the Model Lab's
// sliders redraw cost curves from the champion's test sample without a round trip.

import type { Costs } from "./api";

export type ActionIndex = 0 | 1 | 2 | 3; // APPROVE, REVIEW, BLOCK, STEP_UP (Python order)
export const APPROVE = 0;
export const REVIEW = 1;
export const BLOCK = 2;
export const STEP_UP = 3;

export interface Sample {
  p: number[];
  amount: number[];
  y: number[];
  w: number[];
}

/** Cost of each action if the payment is legit / fraud. */
export function costRow(amount: number, c: Costs): { legit: number[]; fraud: number[] } {
  return {
    legit: [0, c.c_review, c.c_friction, c.c_step_up],
    fraud: [amount, c.c_review + (1 - c.catch_rate) * amount, c.c_block, c.c_step_up + (1 - c.step_up_catch_rate) * amount],
  };
}

/** The cheapest action in expectation (ties -> lower index, as numpy argmin). */
export function bestAction(p: number, amount: number, c: Costs): ActionIndex {
  const { legit, fraud } = costRow(amount, c);
  let best: ActionIndex = APPROVE;
  let bestCost = Infinity;
  const allowed: ActionIndex[] = c.step_up_enabled ? [0, 1, 2, 3] : [0, 1, 2];
  for (const a of allowed) {
    const e = (1 - p) * legit[a] + p * fraud[a];
    if (e < bestCost) {
      bestCost = e;
      best = a;
    }
  }
  return best;
}

function realised(action: ActionIndex, y: number, amount: number, c: Costs): number {
  const { legit, fraud } = costRow(amount, c);
  return y ? fraud[action] : legit[action];
}

/** Weighted total cost of the per-transaction expected-cost rule. */
export function expectedCostRule(s: Sample, c: Costs): number {
  let total = 0;
  for (let i = 0; i < s.p.length; i++) total += s.w[i] * realised(bestAction(s.p[i], s.amount[i], c), s.y[i], s.amount[i], c);
  return total;
}

export interface ThresholdPoint {
  threshold: number;
  cost: number;
  reviewShare: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

/** "Review at or above t, else approve" over a grid of thresholds (weighted counts). */
export function thresholdCurve(s: Sample, c: Costs, grid: number[]): ThresholdPoint[] {
  const total = s.w.reduce((a, b) => a + b, 0);
  return grid.map((t) => {
    let cost = 0;
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;
    for (let i = 0; i < s.p.length; i++) {
      const review = s.p[i] >= t;
      const w = s.w[i];
      cost += w * realised(review ? REVIEW : APPROVE, s.y[i], s.amount[i], c);
      if (s.y[i]) review ? (tp += w) : (fn += w);
      else review ? (fp += w) : (tn += w);
    }
    return { threshold: t, cost, reviewShare: (tp + fp) / total, tp, fp, fn, tn };
  });
}

/** Thresholds spaced on a log scale (p is tiny for most payments). */
export function logGrid(min = 1e-5, max = 0.99, n = 60): number[] {
  const a = Math.log10(min);
  const b = Math.log10(max);
  return Array.from({ length: n }, (_, i) => 10 ** (a + ((b - a) * i) / (n - 1)));
}
