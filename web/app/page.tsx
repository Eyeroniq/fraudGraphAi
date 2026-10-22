"use client";

import Link from "next/link";
import { useMemo } from "react";
import { Histogram, StackedArea } from "@/components/charts";
import { DecisionBadge, Empty, Eyebrow, Glass, Kpi, SectionHead } from "@/components/ui";
import type { Action } from "@/lib/api";
import { bestAction, costRow } from "@/lib/cost";
import { ACTION_META, ACTIONS, count, money, pct, prob } from "@/lib/format";
import { useLiveFeed } from "@/lib/live";
import { useCases, useDrift, useLab } from "@/lib/queries";

const COLORS = Object.fromEntries(ACTIONS.map((a) => [a, ACTION_META[a].color])) as Record<string, string>;
const FALLBACK_COSTS = { c_review: 2000, c_friction: 20000, catch_rate: 0.9, c_block: 0, step_up_enabled: true, c_step_up: 200, step_up_catch_rate: 0.7 };
const INDEX: Record<Action, number> = { APPROVE: 0, REVIEW: 1, BLOCK: 2, STEP_UP: 3 };

export default function LiveMonitor() {
  const live = useLiveFeed();
  const cases = useCases({ status: "open", limit: 1 });
  const drift = useDrift("24h");
  const lab = useLab();
  const costs = lab.data?.policy.costs ?? FALLBACK_COSTS;

  const stats = useMemo(() => {
    const recent = live.recent;
    const flagged = recent.filter((d) => d.decision !== "APPROVE").length;
    const lastMinute = recent.filter((d) => d.at > Date.now() - 60_000).length;
    // Expected loss avoided vs approving everything, under the A6 cost model.
    let saved = 0;
    for (const d of recent) {
      if (d.p_fraud <= 0) continue;
      const { legit, fraud } = costRow(d.amount, costs);
      const e = (a: number) => (1 - d.p_fraud) * legit[a] + d.p_fraud * fraud[a];
      saved += e(0) - e(INDEX[d.decision] ?? bestAction(d.p_fraud, d.amount, costs));
    }
    return { flagged: recent.length ? flagged / recent.length : null, lastMinute, saved };
  }, [live.recent, costs]);

  const rows = live.buckets.map((b) => ({ t: b.t, values: b.counts }));
  const alarms = drift.data?.alarms ?? [];

  return (
    <div className="flex flex-col gap-10">
      <section className="grid items-end gap-10 lg:grid-cols-[1.15fr_1fr]">
        <div className="flex flex-col gap-7">
          <div className="rise" style={{ "--i": 0 } as React.CSSProperties}>
            <Eyebrow live>Real-time risk</Eyebrow>
          </div>
          <h1 className="display rise text-[clamp(3.2rem,9vw,7.5rem)]" style={{ "--i": 1 } as React.CSSProperties}>
            FRAUD
            <br />
            CONSOLE<span className="mint-dot" aria-hidden />
          </h1>
          <p className="rise flex items-center gap-5" style={{ "--i": 2 } as React.CSSProperties}>
            <span aria-hidden className="h-px w-12 bg-[var(--color-mint)]" />
            <span className="label-mono" style={{ color: "var(--text)" }}>
              Payments · graph features · decisions
            </span>
          </p>
          <p className="rise muted max-w-xl text-lg leading-relaxed" style={{ "--i": 3 } as React.CSSProperties}>
            Every payment is scored before authorisation: approved, challenged, held for an analyst or blocked, whichever
            costs least in expectation, with the reasons shown.
          </p>
          <div className="rise flex flex-wrap items-center gap-3" style={{ "--i": 4 } as React.CSSProperties}>
            <Link href="/cases" className="btn btn-primary">
              Review queue <span aria-hidden>↓</span>
            </Link>
            <Link href="/score" className="btn btn-ghost">
              Score a payment <span aria-hidden>↗</span>
            </Link>
            <Link href="/drift" className="btn btn-ghost muted">
              Drift &amp; stream <span aria-hidden>⌁</span>
            </Link>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <Kpi index={2} label="Throughput" value={`${count(stats.lastMinute)}/min`} tip="Decisions received over the live stream in the last minute." sub={`${count(live.total)} since this page opened`} />
          <Kpi index={3} label="Flagged" value={pct(stats.flagged)} tip="Share of recent decisions that were not a plain approval (step-up, review or block)." accent="var(--color-review)" />
          <Kpi index={4} label="Open cases" value={count(cases.data?.total)} tip="Payments held for an analyst that nobody has resolved yet." />
          <Kpi index={5} label="Loss avoided" value={money(stats.saved)} tip="Expected cost of approving these payments minus the expected cost of what was decided, under the cost model (recent decisions)." accent="var(--color-mint)" />
        </div>
      </section>

      {alarms.length > 0 && (
        <Glass index={6} className="flex flex-wrap items-center justify-between gap-4 border-[rgb(251_113_133/0.4)] px-6 py-4" as="div" role="alert">
          <span className="flex items-center gap-3">
            <span className="dot dot-live" style={{ background: "var(--color-block)" }} aria-hidden />
            <span>
              <strong>{alarms.length} drift alarm{alarms.length > 1 ? "s" : ""}</strong>
              <span className="muted"> in the last day · latest: {alarms[alarms.length - 1].kind} ({alarms[alarms.length - 1].detector}) at step {alarms[alarms.length - 1].step}</span>
            </span>
          </span>
          <Link href="/drift" className="btn btn-danger h-10">
            Inspect drift
          </Link>
        </Glass>
      )}

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Glass index={7} className="p-6">
          <SectionHead
            title="Decision mix"
            tip="Decisions per 5-second bucket over the last five minutes, stacked by action."
            right={
              <span className="flex flex-wrap gap-3 text-xs">
                {ACTIONS.map((a) => (
                  <span key={a} className="inline-flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full" style={{ background: COLORS[a] }} aria-hidden />
                    <span className="muted">{ACTION_META[a].label}</span>
                    <span className="num mono">{count(live.counts[a])}</span>
                  </span>
                ))}
              </span>
            }
          />
          {rows.length > 1 ? (
            <StackedArea keys={["APPROVE", "STEP_UP", "REVIEW", "BLOCK"]} colors={COLORS} rows={rows} label="Decisions over time by action" />
          ) : (
            <Empty title="Waiting for traffic">
              Start the replayer on <Link href="/drift" className="text-[var(--color-mint)] underline">Drift &amp; stream</Link>, or score a payment.
            </Empty>
          )}
        </Glass>
        <Glass index={8} className="p-6">
          <SectionHead title="Score distribution" tip="Calibrated fraud probability of recent payments (log-scaled counts so the rare high scores show)." />
          {live.recent.length ? (
            <Histogram values={live.recent.map((d) => d.p_fraud)} label="Histogram of recent fraud probabilities" />
          ) : (
            <Empty title="No scores yet" />
          )}
        </Glass>
      </div>

      <Glass index={9} className="p-6">
        <SectionHead title="Live feed" tip="The latest decisions as they are made (newest first)." right={<span className="faint mono text-xs">{live.status}</span>} />
        {live.recent.length ? (
          <div className="max-h-[28rem] overflow-y-auto">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Transaction</th>
                  <th scope="col">Step</th>
                  <th scope="col">Type</th>
                  <th scope="col" className="text-right">Amount</th>
                  <th scope="col" className="text-right">P(fraud)</th>
                  <th scope="col">Decision</th>
                </tr>
              </thead>
              <tbody>
                {live.recent.slice(0, 60).map((d) => (
                  <tr key={d.txn_id} className="row-in">
                    <td className="mono text-xs">{d.txn_id}</td>
                    <td className="num">{d.step}</td>
                    <td className="muted">{d.type}</td>
                    <td className="num text-right">{money(d.amount)}</td>
                    <td className="num mono text-right text-xs">{prob(d.p_fraud)}</td>
                    <td>
                      <DecisionBadge decision={d.decision} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty title="Nothing streamed yet">The feed fills as soon as payments are scored.</Empty>
        )}
      </Glass>
    </div>
  );
}
