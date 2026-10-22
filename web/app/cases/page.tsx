"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Async, DecisionBadge, Empty, Glass, Meter, PageTitle, Segmented, Tip } from "@/components/ui";
import { age, count, money, prob } from "@/lib/format";
import { useCases } from "@/lib/queries";

const PAGE = 25;
const CAPACITY = 2_000; // reviews per day (Phase 4 policy)
const SLA_MS = 4 * 3600_000;

type Status = "open" | "resolved" | "all";

export default function ReviewQueue() {
  const router = useRouter();
  const [status, setStatus] = useState<Status>("open");
  const [sort, setSort] = useState<"priority" | "created">("priority");
  const [offset, setOffset] = useState(0);
  const [cursor, setCursor] = useState(0);
  const query = useCases({ status: status === "all" ? undefined : status, sort, limit: PAGE, offset });
  const open = useCases({ status: "open", limit: 1 });
  const items = query.data?.items ?? [];

  // J / K move, Enter opens: the queue is worked from the keyboard.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "j") setCursor((c) => Math.min(items.length - 1, c + 1));
      if (e.key === "k") setCursor((c) => Math.max(0, c - 1));
      if (e.key === "Enter" && items[cursor]) router.push(`/cases/view?id=${items[cursor].case_id}`);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [items, cursor, router]);

  const total = query.data?.total ?? 0;
  const openCount = open.data?.total ?? 0;

  return (
    <>
      <PageTitle title="REVIEW QUEUE" subtitle="Held payments, most savings first" />
      <div className="mb-6 grid gap-4 lg:grid-cols-[1fr_22rem]">
        <Glass index={1} className="flex flex-wrap items-center gap-4 p-4" as="div">
          <Segmented<Status>
            label="Case status"
            value={status}
            onChange={(v) => {
              setStatus(v);
              setOffset(0);
              setCursor(0);
            }}
            options={[
              { value: "open", label: "Open" },
              { value: "resolved", label: "Resolved" },
              { value: "all", label: "All" },
            ]}
          />
          <Segmented
            label="Sort order"
            value={sort}
            onChange={(v) => setSort(v)}
            options={[
              { value: "priority", label: "Savings" },
              { value: "created", label: "Newest" },
            ]}
          />
          <span className="faint mono ml-auto hidden text-xs sm:inline">J / K to move · Enter to open</span>
        </Glass>
        <Glass index={2} className="flex flex-col justify-center gap-3 p-5" as="div">
          <div className="flex items-baseline justify-between">
            <Tip tip="Open cases against the team's review capacity per day (2,000 in the policy). Over capacity, the next cheapest action is used instead.">
              <span className="label-mono text-[0.62rem]">Capacity</span>
            </Tip>
            <span className="num mono text-sm">
              {count(openCount)} / {count(CAPACITY)}
            </span>
          </div>
          <Meter value={openCount} max={CAPACITY} label="Open cases against daily capacity" color={openCount > CAPACITY * 0.8 ? "var(--color-review)" : undefined} />
        </Glass>
      </div>

      <Glass index={3} className="p-2 sm:p-4">
        <Async query={query} rows={8} empty={(d) => (d.items.length ? null : <Empty title="No cases here">Held payments appear as the stream runs.</Empty>)}>
          {(data) => (
            <>
              <div className="overflow-x-auto">
                <table className="table">
                  <thead>
                    <tr>
                      <th scope="col">Case</th>
                      <th scope="col" className="text-right">
                        <Tip tip="Expected cost of approving minus expected cost of reviewing: what an analyst's look is worth.">Savings</Tip>
                      </th>
                      <th scope="col">Decision</th>
                      <th scope="col" className="text-right">P(fraud)</th>
                      <th scope="col" className="text-right">Amount</th>
                      <th scope="col">Type</th>
                      <th scope="col">
                        <Tip tip="Time since the case opened; red after the 4-hour review SLA.">Age</Tip>
                      </th>
                      <th scope="col">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.items.map((c, i) => {
                      const late = c.status !== "resolved" && Date.now() - Date.parse(c.created_at) > SLA_MS;
                      return (
                        <tr
                          key={c.case_id}
                          aria-selected={i === cursor}
                          className="cursor-pointer"
                          onClick={() => router.push(`/cases/view?id=${c.case_id}`)}
                        >
                          <td>
                            <Link href={`/cases/view?id=${c.case_id}`} className="mono text-xs hover:text-[var(--color-mint)]">
                              #{c.case_id} · {c.txn_id}
                            </Link>
                          </td>
                          <td className="num text-right font-semibold text-[var(--color-mint)]">{money(c.priority)}</td>
                          <td>
                            <DecisionBadge decision={c.decision} />
                          </td>
                          <td className="num mono text-right text-xs">{prob(c.p_fraud)}</td>
                          <td className="num text-right">{money(c.amount)}</td>
                          <td className="muted">{c.type}</td>
                          <td className={`num mono text-xs ${late ? "text-[var(--color-block)]" : "muted"}`}>
                            {age(c.created_at)}
                            {late && <span className="sr-only"> (past SLA)</span>}
                          </td>
                          <td>
                            {c.status === "resolved" ? (
                              <span className="mono text-xs" style={{ color: c.verdict === "fraud" ? "var(--color-block)" : "var(--color-approve)" }}>
                                {c.verdict}
                              </span>
                            ) : (
                              <span className="mono faint text-xs">{c.status}</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 flex items-center justify-between px-2 text-sm">
                <span className="faint mono text-xs">
                  {offset + 1}–{Math.min(offset + PAGE, total)} of {count(total)}
                </span>
                <span className="flex gap-2">
                  <button type="button" className="btn btn-ghost h-9" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
                    ← Prev
                  </button>
                  <button type="button" className="btn btn-ghost h-9" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>
                    Next →
                  </button>
                </span>
              </div>
            </>
          )}
        </Async>
      </Glass>
    </>
  );
}
