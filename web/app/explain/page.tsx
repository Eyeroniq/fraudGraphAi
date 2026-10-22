"use client";

import { Bars } from "@/components/charts";
import { Async, DecisionBadge, Glass, Kpi, PageTitle, SectionHead } from "@/components/ui";
import type { LabReport } from "@/lib/api";
import { count, pct, prob } from "@/lib/format";
import { useLab } from "@/lib/queries";

const METHODS = [
  { key: "shap", label: "SHAP (ours)", color: "var(--color-mint)" },
  { key: "permutation", label: "Global permutation", color: "var(--color-stepup)" },
  { key: "random", label: "Random", color: "#a1a1aa" },
];
const METRIC_TEXT: Record<string, string> = {
  faithfulness: "Faithfulness: removing high-attribution features moves the score most",
  monotonicity: "Monotonicity: adding features in attribution order raises the score steadily",
  randomisation_corr: "Model randomisation: correlation with a label-shuffled model (lower is better)",
  max_sensitivity: "Max sensitivity: change under tiny input noise (lower is better)",
  complexity: "Complexity: entropy of the attribution (lower is simpler)",
  sparseness: "Sparseness: concentration of the attribution (higher is simpler)",
};

export default function ExplainPage() {
  const lab = useLab();
  return (
    <>
      <PageTitle title="EXPLANATIONS" subtitle="Are the reasons stable and true to the model?" />
      <Async query={lab} rows={8}>
        {(d) => <Body explain={d.explain} />}
      </Async>
    </>
  );
}

function Body({ explain: e }: { explain: LabReport["explain"] }) {
  const metrics = Object.keys(e.quality.shap ?? {});
  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi index={1} label="Top-3 stability" value={e.seed_stability.median_jaccard.toFixed(2)} tip="Median Jaccard overlap of a payment's top-3 reasons across five bootstrap refits of the model (1 = identical)." sub={`${pct(e.seed_stability["share_jaccard_ge_0.8"], 0)} of payments ≥ 0.8`} accent="var(--color-mint)" />
        <Kpi index={2} label="Faithfulness" value={e.quality.shap?.faithfulness?.mean.toFixed(2) ?? "—"} tip="How well attributions predict the score change when features are removed (correlation). Random attributions score about 0." sub={`random ${e.quality.random?.faithfulness?.mean.toFixed(2) ?? "—"}`} />
        <Kpi index={3} label="Flagged unstable" value={pct(e.stability_threshold.share_flagged_unstable, 0)} tip="Share of flagged payments where reasons 3 and 4 are too close to call; the API marks them explanation_stable: false." accent="var(--color-review)" />
        <Kpi index={4} label="Studied" value={count(e.studied.flagged_in_test)} tip="Test payments the policy did not simply approve (the explanations people actually see)." sub={`${count(e.studied.sampled)} sampled`} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Glass index={5} className="p-6">
          <SectionHead title="What drives flags" tip="Mean grouped SHAP contribution (log-odds) on flagged payments: positive pushes towards fraud." />
          <Bars
            fmt={(v) => v.toFixed(2)}
            items={Object.entries(e.flagged_mean_group_shap)
              .filter(([, v]) => Math.abs(v) > 1e-6)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => ({ label: k.replaceAll("_", " ").toLowerCase(), value: v, color: v > 0 ? "var(--color-block)" : "var(--color-mint)" }))}
          />
        </Glass>
        <Glass index={6} className="p-6">
          <SectionHead title="Robustness" tip="How much the top-3 reasons change under input perturbation and a different SHAP background." />
          <Bars
            max={1}
            fmt={(v) => v.toFixed(2)}
            items={[
              { label: "perturbed: same top-3", value: e.perturbation.share_top3_unchanged ?? 0 },
              { label: "perturbed: median Jaccard", value: e.perturbation.median_top3_jaccard ?? 0 },
              { label: "background: mean Jaccard", value: e.background.mean_top3_jaccard ?? 0 },
              { label: "seeds: mean Jaccard", value: e.seed_stability.mean_jaccard },
            ]}
          />
        </Glass>
      </div>

      <Glass index={7} className="p-6">
        <SectionHead title="Quality against baselines" tip="SHAP against global permutation importance and random attributions on the same 500 payments (means; whiskers in the report)." />
        <div className="grid gap-8 md:grid-cols-2 xl:grid-cols-3">
          {metrics.map((m) => (
            <div key={m}>
              <p className="muted mb-3 text-sm">{METRIC_TEXT[m] ?? m}</p>
              <Bars
                fmt={(v) => v.toFixed(2)}
                items={METHODS.filter((x) => e.quality[x.key]?.[m]).map((x) => ({ label: x.label, value: e.quality[x.key][m].mean, color: x.color }))}
              />
            </div>
          ))}
        </div>
      </Glass>

      <Glass index={8} className="p-6">
        <SectionHead title="Examples" tip="Flagged test payments and the reasons the API returned for them." />
        <ul className="grid gap-4 md:grid-cols-2">
          {e.examples.map((x) => (
            <li key={x.txn_id} className="glass flex flex-col gap-3 p-5" style={{ borderRadius: 20 }}>
              <div className="flex items-center justify-between gap-3">
                <span className="mono text-xs">{x.txn_id}</span>
                <span className="flex items-center gap-3">
                  <span className="num mono faint text-xs">p {prob(x.p_fraud)}</span>
                  <DecisionBadge decision={x.action} />
                </span>
              </div>
              <ol className="flex flex-col gap-1.5 text-sm">
                {x.reasons.map((r) => (
                  <li key={r.code} className="flex justify-between gap-3">
                    <span>{r.text}</span>
                    <span className="num mono text-xs text-[var(--color-block)]">+{r.impact.toFixed(2)}</span>
                  </li>
                ))}
              </ol>
            </li>
          ))}
        </ul>
      </Glass>
    </div>
  );
}
