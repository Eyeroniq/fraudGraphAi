"use client";

import { useMemo, useState } from "react";
import { Bars, Heatmap, Lines } from "@/components/charts";
import { useToast } from "@/components/Toaster";
import { Async, Empty, Glass, Meter, PageTitle, SectionHead, Tip } from "@/components/ui";
import type { LabReport, Scenario } from "@/lib/api";
import { count, money, pct } from "@/lib/format";
import { useDrift, useLab, usePerformance, useSim, useSimControl } from "@/lib/queries";

const SCENARIOS: { id: Scenario; text: string }[] = [
  { id: "D0", text: "No drift" },
  { id: "D1", text: "Small-ticket fraud" },
  { id: "D2", text: "Channel shift (gradual)" },
  { id: "D3", text: "Mule rings" },
  { id: "D4", text: "Benign inflation" },
  { id: "D5", text: "Volume surge ×3" },
  { id: "D6", text: "UPI mule VPAs (gradual)" },
];
const POLICIES = [
  { key: "static", label: "Static" },
  { key: "online", label: "Online (river)" },
  { key: "sliding", label: "Daily retrain" },
  { key: "triggered", label: "Alarm retrain" },
  { key: "shadow", label: "Shadow" },
];

export default function DriftPage() {
  return (
    <>
      <PageTitle title="DRIFT & STREAM" subtitle="Replay, inject, watch the monitors" />
      <div className="flex flex-col gap-6">
        <SimControl />
        <LiveMonitors />
        <Study />
      </div>
    </>
  );
}

function SimControl() {
  const sim = useSim();
  const control = useSimControl();
  const toast = useToast();
  const [scenario, setScenario] = useState<Scenario>("D3");
  const s = sim.data;
  const done = s ? Math.max(0, s.next_step - s.first_step) : 0;
  const total = s ? s.last_step - s.first_step + 1 : 1;
  const fail = (e: unknown) => toast(e instanceof Error ? e.message : "Request failed", "error");

  return (
    <Glass index={1} className="p-6">
      <SectionHead
        title="Replayer"
        tip="Streams PaySim through the API step by step (worker), with labels arriving late. Needs an admin key; start and scenario changes are recorded and applied from the next step."
        right={s ? <span className={`mono text-xs ${s.running ? "text-[var(--color-mint)]" : "faint"}`}>{s.running ? "● running" : "○ stopped"}</span> : null}
      />
      <div className="grid items-end gap-6 lg:grid-cols-[1fr_auto]">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Drift scenario">
            {SCENARIOS.map((o) => (
              <button
                key={o.id}
                type="button"
                role="radio"
                aria-checked={scenario === o.id}
                onClick={() => setScenario(o.id)}
                className="pill glass-hover border px-4 py-2 text-sm"
                style={
                  scenario === o.id
                    ? { borderColor: "var(--color-mint)", color: "var(--color-mint)", background: "rgb(110 231 183 / 0.1)" }
                    : { borderColor: "var(--line)", color: "var(--text-muted)" }
                }
              >
                <span className="mono mr-2 text-xs">{o.id}</span>
                {o.text}
              </button>
            ))}
          </div>
          {s && (
            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs">
                <span className="muted">
                  step <span className="num mono text-[var(--text)]">{Math.min(s.next_step, s.last_step)}</span> of {s.first_step}–{s.last_step} · scenario{" "}
                  <span className="mono text-[var(--text)]">{s.scenario}</span> from step {s.scenario_start}
                </span>
                <span className="num mono">{pct(done / total, 0)}</span>
              </div>
              <Meter value={done} max={total} label="Replay progress" />
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            className="btn btn-primary"
            disabled={control.start.isPending}
            onClick={() =>
              control.start.mutate(
                { first_step: 356, last_step: 408, steps_per_tick: 1, scenario: "D0" },
                { onSuccess: () => toast("Replayer started"), onError: fail },
              )
            }
          >
            Start ▸
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={!s?.running}
            onClick={() => control.scenario.mutate(scenario, { onSuccess: () => toast(`${scenario} injected from the next step`), onError: fail })}
          >
            Inject {scenario}
          </button>
          <button type="button" className="btn btn-danger" disabled={!s?.running} onClick={() => control.stop.mutate(undefined, { onSuccess: () => toast("Replayer stopped", "info"), onError: fail })}>
            Stop
          </button>
        </div>
      </div>
    </Glass>
  );
}

function LiveMonitors() {
  const drift = useDrift("1w");
  const perf = usePerformance("1w");
  return (
    <div className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
      <Glass index={2} className="p-6">
        <SectionHead title="Feature drift (PSI)" tip="Population stability index of each feature per day-long window against the stream's first two days. Red cells raised an alarm (significant test and PSI above the tuned threshold)." />
        <Async query={drift} rows={6} empty={(d) => (d.windows.length ? null : <Empty title="No windows yet">The worker writes one after each full day of traffic.</Empty>)}>
          {(d) => {
            const windows = d.windows.filter((w) => w.feature !== "__score__");
            const features = [...new Set(windows.map((w) => w.feature))].sort();
            const ends = [...new Set(windows.map((w) => w.end_step))].sort((a, b) => a - b).slice(-12);
            const at = new Map(windows.map((w) => [`${w.feature}@${w.end_step}`, w]));
            return (
              <Heatmap
                label="PSI per feature and window"
                rows={features}
                cols={ends}
                value={(r, c) => at.get(`${r}@${c}`)?.psi ?? null}
                alarm={(r, c) => at.get(`${r}@${c}`)?.alarm ?? false}
                max={0.5}
                colLabel={(c) => `s${c}`}
              />
            );
          }}
        </Async>
      </Glass>
      <div className="flex flex-col gap-6">
        <Glass index={3} className="p-6">
          <SectionHead title="Alarms" tip="Input (feature PSI), prediction (score drift, alert rate) and performance (missed fraud labels: ADWIN / Page-Hinkley) alarms, newest first." />
          <Async query={drift} rows={4} empty={(d) => (d.alarms.length ? null : <Empty title="All quiet" />)}>
            {(d) => (
              <ol className="flex max-h-72 flex-col gap-2 overflow-y-auto">
                {[...d.alarms].reverse().slice(0, 30).map((a) => (
                  <li key={a.alarm_id} className="flex items-center gap-3 text-sm">
                    <span className="mono faint w-12 text-xs">s{a.step}</span>
                    <span className="pill mono px-2.5 py-0.5 text-[0.65rem] uppercase tracking-widest" style={{ color: a.kind === "performance" ? "var(--color-block)" : a.kind === "prediction" ? "var(--color-review)" : "var(--color-stepup)", border: "1px solid currentColor" }}>
                      {a.kind}
                    </span>
                    <span className="muted truncate">{a.feature ?? a.detector}</span>
                  </li>
                ))}
              </ol>
            )}
          </Async>
        </Glass>
        <Glass index={4} className="p-6">
          <SectionHead title="Champion vs challenger" tip="Precision and recall on labelled payments of the last week. Challengers score in shadow; the worker promotes one that costs less." />
          <Async query={perf} rows={3} empty={(p) => (p.models.length ? null : <Empty title="No labels yet" />)}>
            {(p) => (
              <div className="flex flex-col gap-5">
                {p.models.map((m) => (
                  <div key={m.model_version}>
                    <p className="mb-2 flex items-center justify-between text-xs">
                      <span className="mono truncate">{m.model_version}</span>
                      <span className={m.role === "champion" ? "text-[var(--color-mint)]" : "faint"}>{m.role} · {count(m.n_labelled)} labels</span>
                    </p>
                    <Bars
                      max={1}
                      fmt={(v) => pct(v)}
                      items={[
                        { label: "precision", value: m.precision ?? 0, color: "var(--color-stepup)" },
                        { label: "recall", value: m.recall ?? 0, color: "var(--color-mint)" },
                      ]}
                    />
                  </div>
                ))}
              </div>
            )}
          </Async>
        </Glass>
      </div>
    </div>
  );
}

function Study() {
  const lab = useLab();
  return (
    <Async query={lab} rows={6} empty={(d) => (d.drift.scenarios?.length ? null : <Empty title="No drift study" />)}>
      {(d) => <StudyBody drift={d.drift} />}
    </Async>
  );
}

function StudyBody({ drift }: { drift: LabReport["drift"] }) {
  const [id, setId] = useState("D3");
  const scenario = drift.scenarios.find((s) => s.id === id) ?? drift.scenarios[0];
  const best = useMemo(
    () => Object.fromEntries(drift.scenarios.map((s) => [s.id, Object.entries(s.policies).reduce((a, b) => (b[1].cost_post < a[1].cost_post ? b : a))[0]])),
    [drift.scenarios],
  );
  return (
    <>
      <Glass index={5} className="p-6">
        <SectionHead title="Retraining study (Phase 6)" tip="Cost after the drift starts (step 330) for five policies on each scenario; lower is better, the cheapest in each row is highlighted." />
        <div className="overflow-x-auto">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Scenario</th>
                {POLICIES.map((p) => (
                  <th key={p.key} scope="col" className="text-right">
                    {p.label}
                  </th>
                ))}
                <th scope="col" className="text-right">
                  <Tip tip="Steps from the drift start to the first label-based (performance) alarm. — means it never fired, as it should on harmless drift.">Detected</Tip>
                </th>
              </tr>
            </thead>
            <tbody>
              {drift.scenarios.map((s) => (
                <tr key={s.id} aria-selected={s.id === id} className="cursor-pointer" onClick={() => setId(s.id)}>
                  <td>
                    <span className="mono mr-2 text-xs">{s.id}</span>
                    <span className="muted">{s.kind.replaceAll("_", " ")}</span>
                  </td>
                  {POLICIES.map((p) => {
                    const v = s.policies[p.key];
                    const win = best[s.id] === p.key;
                    return (
                      <td key={p.key} className={`num text-right ${win ? "font-semibold text-[var(--color-mint)]" : ""}`}>
                        {money(v?.cost_post)}
                        <span className="faint mono ml-1.5 text-[0.65rem]">{v ? v.ap_post.toFixed(2) : ""}</span>
                      </td>
                    );
                  })}
                  <td className="num mono text-right text-xs">{s.detection.performance ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="faint mt-3 text-xs">Cost · AP after the drift. Click a row to plot it.</p>
      </Glass>
      <div className="grid gap-6 lg:grid-cols-2">
        <Glass index={6} className="p-6">
          <SectionHead title={`Rolling AP · ${scenario.id}`} tip="Average precision per day-long window for each policy; the dashed line marks the drift start." />
          <Lines
            label={`Rolling AP per policy in scenario ${scenario.id}`}
            series={POLICIES.map((p, i) => ({
              name: p.label,
              color: ["#a1a1aa", "var(--color-review)", "var(--color-mint)", "var(--color-stepup)", "#c4b5fd"][i],
              x: scenario.windows.map((w) => w.start),
              y: scenario.policies[p.key]?.rolling_ap ?? [],
              width: p.key === "sliding" ? 3 : 1.6,
            }))}
            yMin={0}
            yMax={1}
            xFmt={(v) => `s${Math.round(v)}`}
            markers={[{ x: scenario.start_step, label: "drift", color: "var(--color-block)" }]}
          />
        </Glass>
        <Glass index={7} className="p-6">
          <SectionHead title="PaySim's own late period" tip="With no injection, the monitors alarm on every day of steps 409–743: the simulator's volume drifts. Yet the model's daily AP holds, so drift is not damage." />
          <Lines
            label="Natural drift in the late period"
            series={[
              { name: "max feature PSI", color: "var(--color-block)", x: drift.late_period.map((w) => w.start), y: drift.late_period.map((w) => w.max_psi) },
              { name: "score PSI", color: "var(--color-review)", x: drift.late_period.map((w) => w.start), y: drift.late_period.map((w) => w.score_psi) },
              { name: "static AP", color: "var(--color-mint)", x: drift.late_period.map((w) => w.start), y: drift.late_period.map((w) => w.static_ap) },
            ]}
            yMin={0}
            xFmt={(v) => `s${Math.round(v)}`}
          />
        </Glass>
      </div>
    </>
  );
}
