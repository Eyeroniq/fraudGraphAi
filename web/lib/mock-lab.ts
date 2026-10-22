// Synthetic console report for mock builds (same shape as reports/console/lab.json).

import type { LabReport } from "./api";

function rng(seed: number) {
  let a = seed;
  return () => {
    a = (a * 1664525 + 1013904223) >>> 0;
    return a / 4294967296;
  };
}

const curve = (strength: number) => {
  const recall = Array.from({ length: 60 }, (_, i) => i / 59);
  const precision = recall.map((r) => Math.max(0.0017, (1 - r ** (1 / strength)) ** 1.4 * 0.95));
  return { recall, precision };
};

export function lab(): LabReport {
  const r = rng(11);
  const n = 1500;
  const y = Array.from({ length: n }, (_, i) => (i < 120 ? 1 : 0));
  const p = y.map((v) => (v ? 0.05 + 0.9 * r() ** 0.6 : r() ** 9 * 0.3));
  const amount = y.map((v) => Math.round((v ? 300_000 : 60_000) * Math.exp(1.1 * (r() - 0.5) * 2)));
  const w = y.map((v) => (v ? 1 : 250));
  const policy = (total: number, review: number) => ({
    cost: { total, ci95: [total * 0.75, total * 1.3] as [number, number] },
    action_share: { APPROVE: 1 - review - 0.01, STEP_UP: 0.01, REVIEW: review, BLOCK: 0.002 },
  });
  const scen = (id: string, kind: string, costs: number[], aps: number[], detect: (number | null)[]) => ({
    id,
    kind,
    mode: id === "D2" || id === "D6" ? "gradual" : "sudden",
    start_step: 330,
    detection: { input: detect[0], prediction: detect[1], performance: detect[2] },
    policies: Object.fromEntries(
      ["static", "online", "sliding", "triggered", "shadow"].map((name, i) => [
        name,
        {
          ap_post: aps[i],
          cost_post: costs[i] * 1e6,
          rolling_ap: Array.from({ length: 6 }, (_, k) => Math.max(0.05, aps[i] + (r() - 0.5) * 0.15 - (k > 2 ? 0 : 0.05))),
          time_to_recover_steps: i === 2 ? 27 : null,
        },
      ]),
    ),
    windows: Array.from({ length: 6 }, (_, k) => ({ start: 261 + 24 * k, max_psi: 0.06 + 0.07 * k + r() * 0.05, score_psi: r() * 0.05 })),
    alarm_steps: [404],
  });
  const quality = (f: number, rand: number, mono: number) => ({
    faithfulness: { mean: f, ci95: [f - 0.02, f + 0.02] as [number, number], n: 500 },
    monotonicity: { mean: mono, ci95: [mono - 0.01, mono + 0.01] as [number, number], n: 500 },
    randomisation_corr: { mean: rand, ci95: [rand - 0.02, rand + 0.02] as [number, number], n: 500 },
  });
  return {
    generated_at: "2026-10-22T10:00:00+00:00",
    champion: "lgbm_g1",
    models: {
      test: {
        lgbm: { ap: { value: 0.357, ci95: [0.16, 0.6] } },
        logreg: { ap: { value: 0.04, ci95: [0.02, 0.07] } },
      },
    },
    pr_curves: {
      lgbm_g1: { ap: 0.467, prevalence: 0.0017, ...curve(2.4) },
      lgbm: { ap: 0.357, prevalence: 0.0017, ...curve(1.9) },
      lgbm_no_hour: { ap: 0.197, prevalence: 0.0017, ...curve(1.4) },
      logreg: { ap: 0.04, prevalence: 0.0017, ...curve(1.05) },
    },
    sample: { p, amount, y, w, rows: 377_353, fraud: 626 },
    policy: {
      costs: { c_review: 2000, c_friction: 20000, catch_rate: 0.9, c_block: 0, step_up_enabled: true, c_step_up: 200, step_up_catch_rate: 0.7 },
      chosen: { alpha: 0.2, score: "lac", mondrian: true },
      policies_test: {
        global_threshold: policy(153.8e6, 0.029),
        "expected_cost (3 actions)": policy(71.3e6, 0.012),
        "expected_cost + step_up": policy(59.8e6, 0.01),
        "+ conformal (uncertain -> STEP_UP)": policy(66.3e6, 0.011),
      },
      coverage_grid: [0.01, 0.05, 0.1, 0.2, 0.3].map((alpha) => ({
        score: "lac",
        mondrian: true,
        alpha,
        test: { coverage: { overall: 1 - alpha + 0.004, legit: 1 - alpha + 0.005, fraud: 1 - alpha + (r() - 0.5) * 0.04 } },
      })),
      review_load_vs_alpha: [0.05, 0.1, 0.2, 0.3].map((alpha) => ({
        alpha,
        review_share: 0.04 / (alpha * 10),
        reviews_per_day_mean: 4000 / (alpha * 10),
        test_cost: (60 + 30 * alpha) * 1e6,
      })),
      done_when: { target: 0.8, test_fraud_coverage: 0.805 },
    },
    explain: {
      studied: { flagged_in_test: 45226, sampled: 2000, quality_rows: 500 },
      global_mean_abs_group_shap: { DEST_HISTORY: 0.6, PAYMENT_TYPE: 0.34, AMOUNT: 0.32, TIME_OF_DAY: 0.23 },
      flagged_mean_group_shap: { DEST_HISTORY: 1.01, TIME_OF_DAY: 0.57, AMOUNT: 0.41, PAYMENT_TYPE: 0.16 },
      seed_stability: { median_jaccard: 1, mean_jaccard: 0.876, "share_jaccard_ge_0.8": 0.752, median_tau: 0.914 },
      stability_threshold: { threshold: 0.119, balanced_accuracy: 0.725, share_flagged_unstable: 0.44 },
      perturbation: { median_top3_jaccard: 1, share_top3_unchanged: 0.51, median_relative_attribution_change: 0.27 },
      background: { median_top3_jaccard: 1, mean_top3_jaccard: 0.98 },
      quality: { shap: quality(0.64, 0.1, 0.74), permutation: quality(0.34, 0.31, 0.62), random: quality(0.01, 0.02, 0.5) },
      examples: [
        {
          txn_id: "T5116429",
          action: "STEP_UP",
          p_fraud: 0.0004,
          reasons: [
            { code: "AMOUNT", text: "Amount 10,618,211 is in a range where fraud is more common", impact: 1.93 },
            { code: "TIME_OF_DAY", text: "Time of day (20:00) raises the risk score for this payment", impact: 0.61 },
          ],
        },
      ],
    },
    drift: {
      thresholds: { input_psi: 0.32, score_psi: 0.2, alert_z: 4, ph_threshold: 30, adwin_delta: 0.002 },
      scenarios: [
        scen("D0", "none", [91, 231, 82, 92, 91], [0.443, 0.127, 0.508, 0.439, 0.443], [74, null, null]),
        scen("D1", "small_ticket_fraud", [60, 68, 51, 55, 60], [0.235, 0.111, 0.373, 0.304, 0.235], [74, null, 28]),
        scen("D2", "channel_shift", [191, 225, 166, 175, 191], [0.19, 0.091, 0.267, 0.193, 0.19], [74, 26, 27]),
        scen("D3", "mule_ring", [151, 299, 111, 122, 151], [0.216, 0.467, 0.637, 0.467, 0.216], [74, null, 25]),
        scen("D4", "benign_inflation", [109, 242, 105, 111, 109], [0.407, 0.125, 0.473, 0.401, 0.407], [74, null, null]),
        scen("D5", "volume_surge", [150, 304, 129, 134, 150], [0.367, 0.058, 0.452, 0.414, 0.367], [26, null, null]),
        scen("D6", "upi_mule_vpas", [101, 264, 96, 100, 101], [0.223, 0.542, 0.603, 0.398, 0.223], [74, null, 36]),
      ],
      late_period: Array.from({ length: 14 }, (_, k) => ({
        start: 409 + 24 * k,
        fraud_rate: 0.01 + r() * 0.06,
        max_psi: 1.5 + k * 0.33,
        score_psi: 0.28 + k * 0.14,
        static_ap: 0.3 + r() * 0.3,
      })),
    },
  };
}
