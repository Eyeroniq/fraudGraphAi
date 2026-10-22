"use client";

// Case view: the screen an analyst works in. The case ID is a query parameter because a
// static export cannot pre-render pages for IDs that only exist at run time.

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo } from "react";
import { Bars, Waterfall } from "@/components/charts";
import { EgoGraph } from "@/components/EgoGraph";
import { useToast } from "@/components/Toaster";
import { Async, ConformalChip, DecisionBadge, Empty, ErrorState, Glass, Loading, PageTitle, SectionHead, Tip } from "@/components/ui";
import type { Action } from "@/lib/api";
import { ACTION_META, money, prob, REASON_TEXT } from "@/lib/format";
import { useCase, useCases, useExplain, useGraph, useResolve, useTransaction } from "@/lib/queries";

export default function CasePage() {
  return (
    <Suspense fallback={<Loading rows={6} />}>
      <CaseView />
    </Suspense>
  );
}

function CaseView() {
  const params = useSearchParams();
  const router = useRouter();
  const toast = useToast();
  const id = Number(params.get("id"));
  const valid = Number.isInteger(id) && id > 0;
  const kase = useCase(valid ? id : null);
  const txnId = kase.data?.txn_id;
  const txn = useTransaction(txnId);
  const explain = useExplain(txnId);
  const graph = useGraph(txnId);
  const queue = useCases({ status: "open", sort: "priority", limit: 200 });
  const resolve = useResolve();

  const order = useMemo(() => (queue.data?.items ?? []).map((c) => c.case_id), [queue.data]);
  const go = useCallback(
    (delta: number) => {
      const i = order.indexOf(id);
      const next = order[i < 0 ? 0 : i + delta];
      if (next) router.push(`/cases/view?id=${next}`);
    },
    [order, id, router],
  );

  const decide = useCallback(
    (verdict: "fraud" | "legit") => {
      if (!kase.data || kase.data.status === "resolved" || resolve.isPending) return;
      resolve.mutate(
        { id, verdict },
        {
          onSuccess: (r) => {
            toast(`Case #${id} labelled ${verdict}${r.label_created ? "" : " (label already existed)"}`);
            const i = order.indexOf(id);
            const next = order.filter((c) => c !== id)[Math.max(0, i)];
            if (next) router.push(`/cases/view?id=${next}`);
          },
          onError: (e) => toast(e instanceof Error ? e.message : "Could not resolve", "error"),
        },
      );
    },
    [kase.data, resolve, id, toast, order, router],
  );

  // F fraud / L legit / J next / K previous.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.target instanceof HTMLInputElement) return;
      if (e.key === "f") decide("fraud");
      if (e.key === "l") decide("legit");
      if (e.key === "j") go(1);
      if (e.key === "k") go(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [decide, go]);

  if (!valid) return <Empty title="No case selected">Pick one from the <Link className="text-[var(--color-mint)] underline" href="/cases">review queue</Link>.</Empty>;
  if (kase.error) return <ErrorState error={kase.error} retry={() => void kase.refetch()} />;
  if (!kase.data) return <Loading rows={6} />;

  const c = kase.data;
  const pred = txn.data?.prediction;
  const resolved = c.status === "resolved";

  return (
    <>
      <PageTitle title={`CASE #${c.case_id}`} subtitle={`${c.type} · ${money(c.amount)} · step ${c.step}`}>
        <div className="flex flex-wrap items-center gap-3">
          <Link href="/cases" className="btn btn-ghost">
            ← Queue
          </Link>
          <button type="button" className="btn btn-danger" disabled={resolved || resolve.isPending} onClick={() => decide("fraud")} aria-keyshortcuts="f">
            Fraud <kbd className="mono faint text-xs">F</kbd>
          </button>
          <button type="button" className="btn btn-primary" disabled={resolved || resolve.isPending} onClick={() => decide("legit")} aria-keyshortcuts="l">
            Legit <kbd className="mono faint text-xs">L</kbd>
          </button>
          <span className="faint mono text-xs">J / K next · previous</span>
          {resolved && (
            <span className="mono text-sm" style={{ color: c.verdict === "fraud" ? "var(--color-block)" : "var(--color-approve)" }}>
              resolved: {c.verdict}
            </span>
          )}
        </div>
      </PageTitle>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
        <div className="flex flex-col gap-6">
          <Glass index={1} className="p-6">
            <SectionHead title="Decision" tip="What the model decided when the payment arrived, and why that action was cheapest." />
            {pred ? (
              <div className="flex flex-col gap-5">
                <div className="flex flex-wrap items-center gap-4">
                  <DecisionBadge decision={pred.decision} size="lg" />
                  <ConformalChip set={pred.prediction_set} />
                </div>
                <div className="flex items-end gap-6">
                  <div>
                    <Tip tip="Calibrated probability that this payment is fraud.">
                      <span className="label-mono text-[0.62rem]">P(fraud)</span>
                    </Tip>
                    <div className="display num text-5xl">{prob(pred.p_fraud)}</div>
                  </div>
                  <p className="muted pb-1 text-sm">{REASON_TEXT[pred.decision_reason] ?? pred.decision_reason}</p>
                </div>
                <div>
                  <Tip tip="Expected cost of each action for this payment under the cost model; the decision is the cheapest (unless uncertainty or capacity changed it).">
                    <span className="label-mono text-[0.62rem]">Expected cost per action</span>
                  </Tip>
                  <div className="mt-3">
                    <Bars
                      items={Object.entries(pred.expected_cost)
                        .sort((a, b) => a[1] - b[1])
                        .map(([a, v]) => ({ label: ACTION_META[a as Action]?.label ?? a, value: v, color: ACTION_META[a as Action]?.color }))}
                      fmt={(v) => money(v)}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <Loading rows={3} />
            )}
          </Glass>

          <Glass index={2} className="p-6">
            <SectionHead
              title="Why"
              tip="The top reasons that raised the risk, grouped from the model's own attributions (TreeSHAP)."
              right={pred?.explanation_stable === false ? <span className="mono text-xs text-[var(--color-review)]">◐ reasons 3 and 4 nearly tied</span> : null}
            />
            {pred?.reasons.length ? (
              <ol className="flex flex-col gap-3">
                {pred.reasons.map((r, i) => (
                  <li key={r.code} className="flex items-start gap-3">
                    <span className="mono faint pt-0.5 text-xs">{i + 1}</span>
                    <span className="flex-1">{r.text}</span>
                    <span className="num mono text-xs text-[var(--color-block)]">+{r.impact.toFixed(2)}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="muted text-sm">No feature raised the risk; the policy decided.</p>
            )}
          </Glass>

          <Glass index={3} className="p-6">
            <SectionHead title="Attribution" tip="From the model's base margin, each feature pushes the log-odds up (red) or down (mint); the last bar is this payment's margin." />
            <Async query={explain} rows={6}>
              {(e) => (
                <Waterfall
                  bias={e.bias}
                  items={Object.entries(e.contributions)
                    .filter(([, v]) => Math.abs(v) > 1e-4)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .slice(0, 10)
                    .map(([label, value]) => ({ label, value }))}
                  label="SHAP waterfall of the top feature contributions"
                />
              )}
            </Async>
          </Glass>
        </div>

        <div className="flex flex-col gap-6">
          <Glass index={4} className="p-6">
            <SectionHead title="Money graph" tip="Payments within two hops of the sender and receiver, up to this payment's step." />
            <Async query={graph} rows={6} empty={(g) => (g.edges.length ? null : <Empty title="No neighbouring payments" />)}>
              {(g) => <EgoGraph graph={g} focusTxn={c.txn_id} />}
            </Async>
          </Glass>
          <Glass index={5} className="p-6">
            <SectionHead title="Payment" />
            {txn.data ? (
              <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
                {[
                  ["Transaction", txn.data.transaction.txn_id],
                  ["From", txn.data.transaction.name_orig],
                  ["To", txn.data.transaction.name_dest],
                  ["Type", txn.data.transaction.type],
                  ["Amount", money(txn.data.transaction.amount)],
                  ["Model", pred?.model_version ?? "—"],
                ].map(([k, v]) => (
                  <div key={k}>
                    <dt className="label-mono text-[0.6rem]">{k}</dt>
                    <dd className="mono mt-1 truncate text-xs">{v}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <Loading rows={2} />
            )}
          </Glass>
        </div>
      </div>
    </>
  );
}
