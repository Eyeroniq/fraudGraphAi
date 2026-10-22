"use client";

import { useDeferredValue, useMemo, useState } from "react";
import { Lines } from "@/components/charts";
import { Async, Glass, Kpi, PageTitle, SectionHead, Tip } from "@/components/ui";
import type { Costs, LabReport } from "@/lib/api";
import { expectedCostRule, logGrid, thresholdCurve } from "@/lib/cost";
import { count, money, pct } from "@/lib/format";
import { useLab } from "@/lib/queries";

const PALETTE = ["var(--color-mint)", "var(--color-stepup)", "var(--color-review)", "#c4b5fd", "var(--color-block)", "#a1a1aa"];
const NAMES: Record<string, string> = {
  lgbm_g1: "LightGBM + graph (champion)",
  lgbm: "LightGBM",
  lgbm_no_hour: "LightGBM, no hour",
  logreg: "Logistic regression",
  logreg_no_hour: "Logistic, no hour",
  rule_new_dest: "Rule: new destination",
};

export default function ModelLab() {
  const lab = useLab();
  return (
    <>
      <PageTitle title="MODEL LAB" subtitle="Curves, costs and the cheapest threshold" />
      <Async query={lab} rows={10}>
        {(data) => <Lab data={data} />}
      </Async>
    </>
  );
}

function Lab({ data }: { data: LabReport }) {
  const models = Object.keys(data.pr_curves).sort((a, b) => data.pr_curves[b].ap - data.pr_curves[a].ap);
  const champion = data.pr_curves[data.champion];
  const baseline = data.pr_curves.lgbm;
  const gain = champion && baseline ? champion.ap - baseline.ap : null;

  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi index={1} label="Champion AP" value={champion ? champion.ap.toFixed(3) : "—"} tip="Average precision on the test split (steps 356–408): area under the precision-recall curve. Fraud prevalence alone would score about 0.002." accent="var(--color-mint)" />
        <Kpi index={2} label="Test payments" value={count(data.sample.rows)} tip="Modelled payments (transfers and cash-outs) in the test split." sub={`${count(data.sample.fraud)} fraud`} />
        <Kpi index={3} label="Policy cost" value={money(data.policy.policies_test["+ capacity 2000/day, first-come"]?.cost.total)} tip="Realised cost on test of the deployed policy: expected-cost rule, step-up for uncertain payments, 2,000 reviews a day." />
        <Glass index={4} hover as="div" className="flex flex-col justify-between p-5">
          <Tip tip="Pre-registered rule (Phase 3): graph features win only if both the AP and the cost improvement have confidence intervals above zero.">
            <span className="label-mono text-[0.62rem]">Graph verdict</span>
          </Tip>
          <div className="display text-3xl text-[var(--color-mint)]">WIN</div>
          <div className="faint text-xs">{gain !== null ? `+${gain.toFixed(3)} AP over LightGBM without graph` : ""}</div>
        </Glass>
      </div>

      <Glass index={5} className="p-6">
        <SectionHead title="Precision–recall on test" tip="How many flagged payments are fraud (precision) at each share of fraud caught (recall). Higher and further right is better." />
        <Lines
          label="Precision-recall curves per model"
          series={models.map((m, i) => ({
            name: `${NAMES[m] ?? m} · AP ${data.pr_curves[m].ap.toFixed(3)}`,
            color: PALETTE[i % PALETTE.length],
            x: data.pr_curves[m].recall,
            y: data.pr_curves[m].precision,
            width: m === data.champion ? 3 : 1.6,
            dashed: m.startsWith("rule") || m.startsWith("logreg"),
          }))}
          yMin={0}
          yMax={1}
          xLabel="recall"
          yLabel="precision"
          xFmt={(v) => pct(v, 0)}
          yFmt={(v) => pct(v, 0)}
        />
      </Glass>

      <CostLab data={data} />

      <div className="grid gap-6 lg:grid-cols-2">
        <Glass index={8} className="p-6">
          <SectionHead title="Conformal coverage" tip="Share of payments whose true label is inside the prediction set, per class, against the target 1 − α. Mondrian sets keep fraud coverage near the target too." />
          <Lines
            label="Coverage against alpha"
            series={(() => {
              const grid = data.policy.coverage_grid.filter((g) => g.mondrian && g.score === data.policy.chosen.score).sort((a, b) => a.alpha - b.alpha);
              const x = grid.map((g) => g.alpha);
              return [
                { name: "target 1 − α", color: "var(--text-faint)", x, y: x.map((a) => 1 - a), dashed: true },
                { name: "legit", color: "var(--color-approve)", x, y: grid.map((g) => g.test.coverage.legit) },
                { name: "fraud", color: "var(--color-block)", x, y: grid.map((g) => g.test.coverage.fraud) },
              ];
            })()}
            yMin={0}
            yMax={1}
            xLabel="alpha"
            xFmt={(v) => v.toFixed(2)}
            yFmt={(v) => pct(v, 0)}
            markers={[{ x: data.policy.chosen.alpha, label: `chosen α ${data.policy.chosen.alpha}`, color: "var(--color-mint)" }]}
          />
        </Glass>
        <Glass index={9} className="p-6">
          <SectionHead title="Review load vs α" tip="Lower α means bigger sets: more payments are 'uncertain' and need a check. This is the analyst workload that choice implies." />
          <Lines
            label="Reviews per day against alpha"
            series={[
              {
                name: "reviews / day",
                color: "var(--color-review)",
                x: data.policy.review_load_vs_alpha.map((r) => r.alpha),
                y: data.policy.review_load_vs_alpha.map((r) => r.reviews_per_day_mean),
              },
            ]}
            xLabel="alpha"
            xFmt={(v) => v.toFixed(2)}
            yFmt={(v) => count(v)}
          />
        </Glass>
      </div>
    </div>
  );
}

function Slider({ label, tip, value, min, max, step, fmt, onChange }: { label: string; tip: string; value: number; min: number; max: number; step: number; fmt: (v: number) => string; onChange: (v: number) => void }) {
  const id = `s-${label.replace(/\W+/g, "-")}`;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <label htmlFor={id}>
          <Tip tip={tip}>
            <span className="label-mono text-[0.62rem]">{label}</span>
          </Tip>
        </label>
        <span className="num mono text-sm">{fmt(value)}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

function CostLab({ data }: { data: LabReport }) {
  const [costs, setCosts] = useState<Costs>(data.policy.costs);
  const [t, setT] = useState(-2.3); // log10 threshold
  const deferred = useDeferredValue(costs); // keep the sliders smooth while the curve recomputes
  const grid = useMemo(() => logGrid(1e-5, 0.99, 64), []);
  const curve = useMemo(() => thresholdCurve(data.sample, deferred, grid), [data.sample, deferred, grid]);
  const rule = useMemo(() => expectedCostRule(data.sample, deferred), [data.sample, deferred]);
  const best = curve.reduce((a, b) => (b.cost < a.cost ? b : a), curve[0]);
  const threshold = 10 ** t;
  const at = useMemo(() => thresholdCurve(data.sample, deferred, [threshold])[0], [data.sample, deferred, threshold]);
  const approveAll = curve[curve.length - 1];

  return (
    <Glass index={6} className="p-6">
      <SectionHead
        title="Cost of a threshold"
        tip="Cost on the test sample of 'review at or above the threshold, else approve' as the costs change. The dashed line is the per-payment expected-cost rule, which needs no threshold."
        right={
          <button type="button" className="btn btn-ghost h-9 text-sm" onClick={() => setCosts(data.policy.costs)}>
            Reset costs
          </button>
        }
      />
      <div className="grid gap-8 lg:grid-cols-[18rem_1fr]">
        <div className="flex flex-col gap-6">
          <Slider label="Review cost" tip="What one analyst review costs (c_review)." value={costs.c_review} min={100} max={20000} step={100} fmt={money} onChange={(v) => setCosts({ ...costs, c_review: v })} />
          <Slider label="Block friction" tip="Cost of wrongly blocking a legit customer (c_friction)." value={costs.c_friction} min={1000} max={200000} step={1000} fmt={money} onChange={(v) => setCosts({ ...costs, c_friction: v })} />
          <Slider label="Catch rate" tip="Share of reviewed fraud an analyst actually stops (r)." value={costs.catch_rate} min={0.5} max={1} step={0.01} fmt={(v) => pct(v, 0)} onChange={(v) => setCosts({ ...costs, catch_rate: v })} />
          <Slider label="Threshold" tip="Review payments with calibrated P(fraud) at or above this." value={t} min={-5} max={-0.01} step={0.01} fmt={(v) => (10 ** v).toExponential(1)} onChange={setT} />
          <dl className="grid grid-cols-2 gap-3 text-sm">
            {[
              ["Caught", count(at.tp)],
              ["Missed", count(at.fn)],
              ["False alarms", count(at.fp)],
              ["Review share", pct(at.reviewShare)],
            ].map(([k, v]) => (
              <div key={k} className="glass p-3" style={{ borderRadius: 16 }}>
                <dt className="label-mono text-[0.58rem]">{k}</dt>
                <dd className="num mt-1 font-semibold">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div>
          <Lines
            label="Total cost against review threshold"
            series={[
              { name: "threshold policy", color: "var(--color-mint)", x: curve.map((c) => c.threshold), y: curve.map((c) => c.cost) },
              { name: "expected-cost rule", color: "var(--color-stepup)", x: [grid[0], grid[grid.length - 1]], y: [rule, rule], dashed: true },
            ]}
            xLog
            xLabel="review threshold (log)"
            yLabel="cost"
            xFmt={(v) => v.toExponential(0)}
            yFmt={money}
            markers={[
              { x: best.threshold, label: `cheapest ${money(best.cost)}`, color: "var(--color-mint)" },
              { x: threshold, label: "yours", color: "var(--color-review)" },
            ]}
          />
          <p className="muted mt-4 text-sm">
            At your threshold: <strong className="num text-[var(--text)]">{money(at.cost)}</strong> · approving everything:{" "}
            <span className="num">{money(approveAll.cost)}</span> · per-payment rule: <span className="num text-[var(--color-stepup)]">{money(rule)}</span>
          </p>
        </div>
      </div>
    </Glass>
  );
}
