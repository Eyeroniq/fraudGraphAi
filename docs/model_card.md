# Model card: FraudGraph champion `lgbm_g1`

Attached to every release (`.github/workflows/release.yml`). The console shows the same card
live at `/model`. Every number here comes from the reports named in each section.

## What it is

- **Model:** LightGBM, 102 trees.
- **Inputs:** decision-time features from the FraudGraph feature engine (version 1.3.0):
  - transaction features (amount, type, hour);
  - sender history;
  - receiving-account graph features (G1: destination novelty, age, incoming volume, distinct
    senders, sender–receiver pair history).
- **Output:** an isotonic-calibrated fraud probability. The decision policy turns it into
  one of four actions.
- **Policy:** the cheapest action in expectation under the A6 cost model, with these costs:

  | Action | Cost |
  |---|---|
  | review | 2,000 |
  | wrongly blocking a legit customer | 20,000 |
  | step-up (OTP / 3DS) | 200 |

  - Analysts stop 90 % of reviewed fraud; a step-up challenge stops 70 %.
  - Payments that conformal prediction marks uncertain (α = 0.2, class-conditional sets) get at
    least a step-up.
  - Reviews are capped at 2,000 a day.

## Intended use

Pre-authorisation risk scoring of transfers and cash-outs on a mobile-money network. Each
payment is approved, challenged, held for an analyst or blocked, and comes with reasons.
Other payment types are approved without scoring. The model is not meant for credit
decisions, account closure without human review, or populations unlike PaySim's.

## Data

**Source:** PaySim, a mobile-money simulator: 6,362,620 payments, 8,213 fraud, 743 hourly
steps. It is pinned by sha256 ([data_audit.md](data_audit.md)).

**Splits**, all by time, with embargoes the length of the label delay:

| Split | Steps |
|---|---|
| train | 1–236 |
| validation | 261–280 |
| calibration | 305–355 |
| test | 356–408 |
| late test | 409–743 |

**Excluded inputs:** sender and receiver balances. In PaySim, fraud empties the sender's
account exactly, so `amount == oldbalance_orig` alone scores AP 0.98. That is a leak of the
simulator, not a signal a real system would have.

## Performance (test, steps 356–408)

| Measure | Value | Source |
|---|---|---|
| Average precision | **0.467** (fraud rate 0.17 %) | `fgx validate`, reports/console |
| AP without graph features (LightGBM) | 0.357 | [graph_study.md](graph_study.md) |
| AP without hour of day | 0.197 | Phase 1 ablation |
| Realised cost, full policy | **66.3 M** vs 153.8 M for one global threshold | [conformal.md](conformal.md) |
| Fraud coverage of conformal sets | 0.805 at α = 0.2 | [conformal.md](conformal.md) |
| Calibration skill (Brier vs constant) | 0.31 | `fgx validate` |
| Explanation faithfulness | 0.64 (random 0.01) | [explain_quality.md](explain_quality.md) |

On the pre-registered rule, graph features **win**: both the AP gain and the cost saving have
confidence intervals above zero ([graph_study.md](graph_study.md)).

## Known limits and findings

- **Weak segments:**
  - On the three lower amount quartiles, test AP is 0.23 against 0.47 overall (`fgx validate`
    warning).
  - Small fraud is harder, which is why drift scenario D1 (small-ticket fraud) hurts.
- **Unused features:**
  - `o_amount_z` is empty on all PaySim rows, and `g_pair_n` is constant.
  - Both carry no signal here (validation warnings). They are kept for data with repeat
    senders.
- **Calibration ties:**
  - Isotonic calibration is a step function: ranking by the calibrated probability gives AP
    0.415 instead of 0.467.
  - The review queue orders by expected savings (probability × amount), which breaks most
    ties.
- **Hour of day** is unusually predictive in the simulator. Reasons show it, so analysts can
  discount it.
- **New fraud patterns** (mule rings, UPI mule VPAs) are missed until labels arrive.
  - The daily retrain recovers within about a day ([drift_study.md](drift_study.md)).
  - The label-based monitor detects them 25–36 steps after they start.
- **Small test set:** fraud is rare (626 frauds in test), so confidence intervals are wide
  (AP about 0.29–0.63).
- **No fairness analysis:** the customers are simulated, so no real-world fairness or
  demographic analysis is possible.

## Operation

- **Serving:** a verified bundle (hashed files, engine-version guard) behind the FastAPI
  service, with p95 under 50 ms at 100 rps on two workers on a laptop.
- **Retraining:** daily, into a shadow challenger, which is promoted only when its realised
  cost on labelled traffic is lower ([operations.md](operations.md)).
- **Release gates:** data integrity, train/test validation, model evaluation, and behaviour
  tests (invariance, directional, minimum functionality, floors, conformal coverage).
