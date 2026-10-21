# Cost model, calibration and decision policy

Task 1.5. Produced by `uv run fgx policy` for the champion (`lgbm`, from `configs/base.yaml`);
numbers below are on the **test** split (steps 356–408, 377,353 TRANSFER + CASH_OUT rows,
626 fraud), with 95 % block-bootstrap CIs (6-step blocks, 1,000 resamples). Full output:
`reports/eval/policy_lgbm/policy_report.json`, artifacts: `artifacts/policy/lgbm/`.

## Verdict

The per-transaction **expected-cost rule** reviews 4.4 % of transactions, stops 93.8 % of the
fraud amount, and costs **130 M** against **1,082 M** for approving everything — 88 % less.
The classic single global threshold reviews a similar 4.0 % but costs **218 M**, because it
ignores the amount: it spends reviews on small transfers and lets large ones through.

| Rule (test) | Total cost | 95 % CI | Review | Block | Fraud amount stopped |
|---|---:|---|---:|---:|---:|
| Approve all | 1,082 M | [669 M, 1,578 M] | 0 % | 0 % | 0 % |
| Global threshold (alert → review) | 218 M | [174 M, 270 M] | 3.97 % | 0 % | 91.8 % |
| **Expected-cost rule** | **130 M** | [87 M, 170 M] | 4.42 % | 0.06 % | 93.8 % |
| Expected-cost, prior-shifted to test prevalence | 132 M | [89 M, 172 M] | 5.46 % | 0.08 % | 94.4 % |

## Cost parameters: why not 5 / 50

A6 was written with `c_review = 5` and `c_friction = 50`. PaySim amounts are large (median legit
TRANSFER/CASH_OUT ≈ 168k, median fraud ≈ 441k), so at 5 per review reviewing is almost free:
the global threshold then reviews **99.3 %** of transactions and the expected-cost rule reviews
71 % and blocks 18 %. Neither is a policy anyone could run.

The costs only mean something on the same scale as the amounts. We set **`c_review = 2,000`**
(≈ 1 % of a median legit transfer: analyst time plus customer delay) and **`c_friction = 20,000`**
(≈ 12 %: churn and complaint cost of wrongly blocking a customer). `catch_rate = 0.9`,
`c_block = 0` are unchanged.

| `c_review` / `c_friction` | Global: review | Global cost | Expected-cost: review | Expected-cost cost | Stopped |
|---|---:|---:|---:|---:|---:|
| 5 / 50 (A6 original) | 99.3 % | 110 M | 71.0 % (+18 % block) | 7.7 M | 99.9 % |
| 500 / 5,000 | 22.3 % | 170 M | 15.3 % | 85 M | 97.2 % |
| **2,000 / 20,000** | **4.0 %** | **218 M** | **4.4 %** | **130 M** | **93.8 %** |
| 5,000 / 50,000 | 4.0 % | 263 M | 2.2 % | 179 M | 89.7 % |
| 20,000 / 200,000 | 0.3 % | 333 M | 0.6 % | 229 M | 85.4 % |

Review capacity (A6 step 3, task 4.2) is the real constraint; these defaults put the load in a
range a team could plausibly staff, and 4.2 will make the capacity explicit.

## Sensitivity (each parameter ×0.5 / ×2, the others fixed)

| Parameter | ×0.5 | ×1 | ×2 |
|---|---|---|---|
| `c_review` | EC 110 M, review 7.8 % | EC 130 M, review 4.4 % | EC 162 M, review 2.5 % |
| `c_friction` | EC 128 M | EC 130 M | EC 131 M |
| `catch_rate` | EC 206 M, global 665 M | EC 130 M, global 218 M | — (capped at 1) |

The result is driven by review cost and catch rate; friction barely matters because blocking
is rare (only very high-risk, high-amount rows).

## Calibration

Isotonic regression fitted on `calib_a` only (Platt as the comparison):

| Split | Raw Brier / ECE | Isotonic Brier / ECE | Platt Brier / ECE |
|---|---|---|---|
| calib_b | 0.00101 / 0.00068 | 0.00109 / 0.00049 | 0.00119 / 0.00074 |
| test | 0.00125 / 0.00054 | 0.00130 / 0.00048 | 0.00151 / 0.00094 |
| late_test | 0.01710 / 0.01488 | 0.01904 / 0.01802 | 0.02168 / 0.02144 |

The champion was trained without class weighting, so its raw scores are already close to
calibrated; isotonic lowers ECE slightly on calib_b/test at a tiny Brier cost. On the late
period every version is badly calibrated: prevalence there is 2.3 % against 0.14 % in calib_a.

## Prior shift

Re-weighting the calibrated probabilities from the calib_a prevalence (0.137 %) to the test
prevalence (0.166 %) barely changes the result (132 M vs 130 M, CIs overlap fully). On test the
prevalence gap is small; the rule matters much more under the late-period shift, where
prevalence is 17× higher (task 4.3 and Phase 6).
