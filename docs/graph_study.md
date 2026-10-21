# Graph study: did the graph help?

Task 3.6. Produced by `uv run fgx graph-study` (full output: `reports/graph_study/results.json`).
Judged by the rule pre-registered in the roadmap (3.0, frozen in commit `09e2edd` before any
graph result): over 3 seeds (scores averaged per row), a graph variant **wins** if the paired
block-bootstrap 95 % CI of ΔAP is above 0 **and** of Δ test cost is below 0, both vs the
baseline E0, **and** its p95 latency fits the 50 ms budget. E2 and E3 are secondary
comparisons with a Holm correction.

## Verdict

**Yes, the hand-built graph features win on PaySim; the GNN adds no robust gain beyond
them; and the graph model does not learn mule rings it has never seen.**

- **E1 (tabular + G1 graph features) wins**: ΔAP **+0.133 [+0.088, +0.157]**, Δcost
  **−66 M [−86 M, −47 M]** (test cost halves, 133 M → 66 M), p95 latency 0.12 ms.
  The gain comes from **G1-a, the receiving account's earlier activity** (novelty, recent
  incoming payments, distinct senders): removing it removes the gain.
- **E1 without G1-b also wins** (ΔAP +0.122 [+0.083, +0.138], Δcost −66 M). G1-b needs labels
  online, which the API only gets in 7.3, so *this* is the servable graph model today.
- **E2 (hybrid with GNN embeddings) also passes the rule, but does not beat E1**: slightly
  higher test AP (0.484 vs 0.457) yet higher cost (73 M vs 66 M) and a much worse late
  period (0.345 vs 0.477), at 8× the latency. **E3 (GNN end-to-end) is worse than the
  baseline** (ΔAP −0.18). The GNN adds no robust gain over the hand-built destination features.
- **E4 (injected mule rings): the graph model is worse at catching rings than the
  baseline** (ring AP 0.013 vs 0.105). PaySim has almost no rings, so E1 learned
  "many recent senders = a busy, legitimate account" — exactly what a ring looks like.

## Setup

- Test: steps 356–408, 377,353 TRANSFER + CASH_OUT rows, 626 fraud, 9 six-step blocks (CIs
  are wide: the dense test window is only 2.2 days). Late test: steps 409–743 (prevalence 2.3 %).
- LightGBM runs: 20-trial seeded search on `train_val` per seed; GNN: causal neighbourhood
  (last 10 earlier edges into the destination), GINE chosen over SAGE on `train_val` AP
  (0.272 vs 0.267).
- Cost: each model gets its own isotonic calibration on `calib_a` and the expected-cost rule
  (costs from `configs/base.yaml`, see [policy.md](policy.md)).

## Results (seed-averaged scores)

| Run | Features | Test AP | Late AP | Test cost | ΔAP vs E0 [95 % CI] | Δcost vs E0 [95 % CI] |
|---|---|---:|---:|---:|---|---|
| **E0** | T+O (baseline) | 0.324 | 0.455 | 133 M | — | — |
| **E1** | T+O+G1 | **0.457** | **0.477** | **66 M** | **+0.133 [+0.088, +0.157]** | **−66 M [−86, −47]** |
| E2 | T+O+G1+G2 (GNN emb.) | 0.484 | 0.345 | 73 M | +0.160 [+0.090, +0.223] (Holm p < 0.001) | −60 M [−75, −42] (Holm p < 0.001) |
| E3 | GNN end-to-end | 0.144 | 0.232 | 115 M | −0.180 [−0.277, −0.070] | −17 M [−34, +2] (Holm p = 0.09) |
| E5a | E0 + sender balance | 1.000 | 1.000 | 0.06 M | artifact | artifact |
| E5b | E0 + all balances | 1.000 | 0.999 | 0.02 M | artifact | artifact |

`T` transaction, `O` origin history, `G1` hand-built graph features, `G2` GNN edge
embeddings. E5 runs are artifact references only (D2): one balance comparison separates
fraud perfectly, which is why balances are excluded.

**Verdicts by the rule:** E1 **WIN**, E2 **WIN** (but not better than E1: higher cost, worse
late period), E3 no significant gain (Holm).

## Ablations (single seed)

| Run | Test AP | Test cost | Reading |
|---|---:|---:|---|
| E1 | 0.457 | 66 M | |
| E1 − G1-a (destination activity) | 0.310 | 147 M | the whole gain is here |
| E1 − G1-b (= E1 without G1-b) | 0.446 | 67 M | barely matters (and needs online labels) |
| E1 − G1-c (pair history) | 0.456 | 70 M | nothing: senders never repeat |
| E0 without hour | 0.204 | 187 M | |
| E1 without hour | 0.235 | 93 M | graph still halves cost; AP gain shrinks |

Hour (the simulator-inflated feature from the data audit) interacts with the graph features:
with hour, the AP gain is +0.13; without it, the point estimate is +0.03, while the cost still
halves (187 M → 93 M). Read the AP gain as partly hour-assisted; the cost gain holds either way.

Where LightGBM spends its gain (E1, seed 0): transaction 77 %, graph G1 23 %, origin 0 %.
In the hybrid E2, GNN embeddings take 35 % of the gain (G1 drops to 10 %): the embeddings
largely re-encode what G1 already measures, and the result does not hold up on the late period.

## E4: injected mule rings

20 rings × 3 new destinations × 15 new senders, each ring within a 6-step burst, injected
into the test period (900 fraud edges that do not drain the sender). Models are the ones
trained on clean data; nothing is retrained.

| Model | Test AP | Ring AP | Ring edges in top 1 % |
|---|---:|---:|---:|
| E0 | 0.304 | 0.105 | 34 % |
| E1 | 0.241 | 0.013 | 18 % |

ΔAP E1 − E0 on the injected test: −0.06 [−0.22, +0.04].

This is the most important negative result. A graph model only knows the graph patterns in
its training labels. PaySim's fraud goes to one-off destinations (44 destinations ever
receive more than one fraud), so E1 learned that a destination with several recent senders
is a busy, normal account. A mule ring creates exactly that. Detecting rings needs
labelled ring examples reaching the model (delayed labels → online learning, Phase 6), or
an explicit burst-into-a-brand-new-account signal checked under the D3 scenario in 6.2.

## Latency (p95 per transaction, engine features + model, CPU)

| Model | p95 |
|---|---:|
| E0 / E1 (LightGBM) | 0.12 ms |
| E3 (causal neighbourhood + GNN forward) | 0.82 ms |
| E2 (E1 + GNN embedding) | 0.97 ms |

All far inside the 50 ms budget. The GNN would be servable: its neighbourhood is causal
and small, so it can be built per request from engine state, with no stale embeddings.

## Caveats

- Nine test blocks: the CIs are honest but wide; the late test (56 blocks) agrees on E1 > E0
  (0.477 vs 0.455) but less strongly.
- E2's training rows use strictly out-of-fold embeddings with a 24-step label embargo before
  each fold. Its `train_val` embeddings come from a GNN early-stopped on `train_val` (epoch
  choice only), so E2's validation AP is slightly optimistic.
- A code review before publishing found and fixed: the missing embargo in the out-of-fold
  embeddings, Holm applied to ΔAP only (now to Δcost too), a duplicated ablation run and two
  wasted GNN fits. The study was re-run; only E2 and E3 numbers moved.
- Seed spread is real: E0's test AP is 0.324 here (3-seed average) vs 0.357 for the single
  seed-42 baseline in 1.3.
- PaySim is simulated. The direction (destination novelty helps) matches how mule accounts
  behave in practice; the size of the gain should not be expected elsewhere.

## Decision

- Champion candidate: **E1 without G1-b** (`lgbm` on T+O+G1 minus `g_dest_known_fraud_in`),
  servable today; retrain and bundle it in 7.5 (champion/challenger). Add G1-b back once
  `/v1/labels` exists (7.3).
- Keep the GNN as a research result: no robust gain over hand-built features, ~8× the latency.
- Re-run E4 with the drift scenario library (6.2) and online learning (6.3): the ring
  failure is the case the streaming phase must fix.
