"use client";

import { type FormEvent, useState } from "react";
import { Bars } from "@/components/charts";
import { ConformalChip, DecisionBadge, Glass, PageTitle, SectionHead, Tip } from "@/components/ui";
import { type Action, api, type ScoreResponse, type TxnType } from "@/lib/api";
import { ACTION_META, money, ms, prob, REASON_TEXT } from "@/lib/format";

const TYPES: TxnType[] = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"];
const newId = () => `W${Date.now().toString().slice(-8)}`;

export default function ScorePage() {
  const [form, setForm] = useState({
    txn_id: newId(),
    step: 360,
    type: "TRANSFER" as TxnType,
    amount: 181000,
    name_orig: "C1305486145",
    name_dest: "C553264065",
  });
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const set = (key: keyof typeof form, value: string) =>
    setForm((f) => ({ ...f, [key]: key === "step" || key === "amount" ? Number(value) : value }));

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await api.score(form));
      setForm((f) => ({ ...f, txn_id: newId() })); // the next submit is a new payment
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const field = (key: keyof typeof form, label: string, props: React.InputHTMLAttributes<HTMLInputElement> = {}) => (
    <label className="flex flex-col gap-2">
      <span className="label-mono text-[0.6rem]">{label}</span>
      <input className="field num" value={form[key]} onChange={(e) => set(key, e.target.value)} required {...props} />
    </label>
  );

  return (
    <>
      <PageTitle title="SCORE A PAYMENT" subtitle="One request, one decision, the reasons" />
      <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        <Glass index={1} className="p-6">
          <form onSubmit={submit} className="grid gap-5 sm:grid-cols-2">
            {field("txn_id", "Transaction ID")}
            {field("step", "Step (hour 1–744)", { type: "number", min: 1, max: 744 })}
            <label className="flex flex-col gap-2">
              <span className="label-mono text-[0.6rem]">Type</span>
              <select className="field" value={form.type} onChange={(e) => set("type", e.target.value)}>
                {TYPES.map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
            </label>
            {field("amount", "Amount", { type: "number", min: 0, step: "0.01" })}
            {field("name_orig", "From account", { pattern: "[CM]\\d+" })}
            {field("name_dest", "To account", { pattern: "[CM]\\d+" })}
            <div className="flex items-center gap-4 sm:col-span-2">
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {busy ? "Scoring…" : "Score"} <span aria-hidden>→</span>
              </button>
              {error && (
                <p role="alert" className="text-sm text-[var(--color-block)]">
                  {error}
                </p>
              )}
            </div>
          </form>
        </Glass>
        <Glass index={2} className="p-6" aria-live="polite">
          <SectionHead title="Decision" />
          {result ? (
            <div className="flex flex-col gap-6">
              <div className="flex flex-wrap items-center gap-4">
                <DecisionBadge decision={result.decision} size="lg" />
                <ConformalChip set={result.prediction_set} />
                <span className="faint mono ml-auto text-xs">{ms(result.latency_ms)} · {result.model_version}</span>
              </div>
              <div className="flex items-end gap-6">
                <div>
                  <span className="label-mono text-[0.6rem]">P(fraud)</span>
                  <div className="display num text-5xl">{prob(result.p_fraud)}</div>
                </div>
                <p className="muted pb-1 text-sm">{REASON_TEXT[result.decision_reason] ?? result.decision_reason}</p>
              </div>
              <div>
                <Tip tip="Expected cost of each action for this payment: the decision is the cheapest one.">
                  <span className="label-mono text-[0.6rem]">Expected cost</span>
                </Tip>
                <div className="mt-3">
                  <Bars
                    fmt={money}
                    items={Object.entries(result.expected_cost)
                      .sort((a, b) => a[1] - b[1])
                      .map(([a, v]) => ({ label: ACTION_META[a as Action]?.label ?? a, value: v, color: ACTION_META[a as Action]?.color }))}
                  />
                </div>
              </div>
              {result.reasons.length > 0 && (
                <ol className="flex flex-col gap-2 text-sm">
                  {result.reasons.map((r) => (
                    <li key={r.code} className="flex justify-between gap-3">
                      <span>{r.text}</span>
                      <span className="num mono text-xs text-[var(--color-block)]">+{r.impact.toFixed(2)}</span>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          ) : (
            <p className="muted text-sm">Fill in a payment and score it: the decision, its probability and the reasons appear here.</p>
          )}
        </Glass>
      </div>
    </>
  );
}
