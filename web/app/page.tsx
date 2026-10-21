"use client";

import { type FormEvent, useState } from "react";
import { type Action, score, type ScoreResponse, type TxnType } from "@/lib/api";

const TYPES: TxnType[] = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"];
const ICON: Record<Action, string> = { APPROVE: "✓", REVIEW: "⚑", BLOCK: "✕" };

function randomId(): string {
  return `W${Date.now().toString().slice(-7)}`;
}

export default function ScorePage() {
  const [form, setForm] = useState({
    txn_id: randomId(),
    step: 360,
    type: "TRANSFER" as TxnType,
    amount: 181000,
    name_orig: "C1305486145",
    name_dest: "C553264065",
  });
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const update = (key: keyof typeof form) => (value: string) =>
    setForm((f) => ({ ...f, [key]: key === "step" || key === "amount" ? Number(value) : value }));

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await score(form));
      setForm((f) => ({ ...f, txn_id: randomId() })); // next submit is a new transaction
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <h1>FraudGraph</h1>
      <p className="muted">Score one payment transaction and see the decision and why it was cheapest.</p>

      <form onSubmit={submit}>
        <div className="grid">
          <label>
            Transaction ID
            <input value={form.txn_id} onChange={(e) => update("txn_id")(e.target.value)} required />
          </label>
          <label>
            Step (hour, 1–744)
            <input
              type="number"
              min={1}
              max={744}
              value={form.step}
              onChange={(e) => update("step")(e.target.value)}
              required
            />
          </label>
          <label>
            Type
            <select value={form.type} onChange={(e) => update("type")(e.target.value)}>
              {TYPES.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
          <label>
            Amount
            <input
              type="number"
              min={0}
              step="0.01"
              value={form.amount}
              onChange={(e) => update("amount")(e.target.value)}
              required
            />
          </label>
          <label>
            Sender account
            <input value={form.name_orig} onChange={(e) => update("name_orig")(e.target.value)} required />
          </label>
          <label>
            Receiving account
            <input value={form.name_dest} onChange={(e) => update("name_dest")(e.target.value)} required />
          </label>
        </div>
        <button type="submit" disabled={busy}>
          {busy ? "Scoring…" : "Score transaction"}
        </button>
      </form>

      {error && (
        <div className="result error" role="alert">
          Could not score: {error}
        </div>
      )}

      {result && (
        <section className="result" aria-live="polite">
          <div className={`decision ${result.decision}`}>
            {ICON[result.decision]} {result.decision}
          </div>
          <p className="muted">
            Fraud probability {(result.p_fraud * 100).toFixed(3)} % · rule {result.decision_reason} · model{" "}
            {result.model_version} · {result.latency_ms.toFixed(1)} ms
          </p>
          <table>
            <tbody>
              {(Object.keys(result.expected_cost) as Action[]).map((action) => (
                <tr key={action}>
                  <td>
                    Expected cost if {action.toLowerCase()}
                    {action === result.decision ? " (chosen: lowest)" : ""}
                  </td>
                  <td>{result.expected_cost[action].toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </main>
  );
}
