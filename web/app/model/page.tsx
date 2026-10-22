"use client";

import { useToast } from "@/components/Toaster";
import { Async, Glass, PageTitle, SectionHead, Tip } from "@/components/ui";
import { useCurrentModel, useModels, usePromote } from "@/lib/queries";

const STAGE: Record<string, string> = {
  champion: "var(--color-mint)",
  challenger: "var(--color-stepup)",
  archived: "var(--text-faint)",
};

export default function ModelCard() {
  const current = useCurrentModel();
  const models = useModels();
  const promote = usePromote();
  const toast = useToast();

  return (
    <>
      <PageTitle title="MODEL CARD" subtitle="What is serving, what it is for, where it fails" />
      <div className="grid gap-6 xl:grid-cols-[1.2fr_1fr]">
        <div className="flex flex-col gap-6">
          <Glass index={1} className="p-6">
            <SectionHead title="Serving now" />
            <Async query={current} rows={4}>
              {(m) => (
                <dl className="grid gap-x-8 gap-y-4 sm:grid-cols-2">
                  {[
                    ["Version", m.version],
                    ["Model", m.model],
                    ["Built", m.created_at ? new Date(m.created_at).toLocaleString() : "—"],
                    ["Feature engine", m.engine_version],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <dt className="label-mono text-[0.6rem]">{k}</dt>
                      <dd className="mono mt-1 break-all text-sm">{v}</dd>
                    </div>
                  ))}
                  <div className="sm:col-span-2">
                    <dt className="label-mono text-[0.6rem]">Features ({m.features.length})</dt>
                    <dd className="mt-2 flex flex-wrap gap-2">
                      {m.features.map((f) => (
                        <span key={f} className="pill mono border border-[var(--line)] px-2.5 py-1 text-[0.7rem]">
                          {f}
                        </span>
                      ))}
                    </dd>
                  </div>
                </dl>
              )}
            </Async>
          </Glass>

          <Glass index={2} className="p-6">
            <SectionHead title="Intended use" />
            <div className="muted flex flex-col gap-3 text-[0.95rem] leading-relaxed">
              <p>
                Pre-authorisation risk for transfers and cash-outs on a mobile-money network: approve, challenge (OTP / 3DS),
                hold for review or block, whichever costs least in expectation. Other payment types are approved unscored.
              </p>
              <p>Trained on PaySim (a simulator), steps 1–236, with labels known only after a 24-step delay.</p>
            </div>
          </Glass>

          <Glass index={3} className="p-6">
            <SectionHead title="Limits & known artifacts" />
            <ul className="muted flex list-disc flex-col gap-2 pl-5 text-[0.95rem] leading-relaxed">
              <li>
                <strong className="text-[var(--text)]">Balance leak excluded.</strong> In PaySim, fraud empties the sender&apos;s balance exactly
                (amount = old balance alone scores AP 0.98). Balances are never model inputs.
              </li>
              <li>
                <strong className="text-[var(--text)]">Hour of day</strong> is unusually predictive in the simulator; the reasons show it so
                analysts can discount it.
              </li>
              <li>Fraud is ~0.2 % of payments: small test sets give wide confidence intervals (AP 0.29–0.63).</li>
              <li>New fraud patterns (mule rings) are missed until labels arrive; the daily retrain recovers within about a day.</li>
              <li>Simulated customers: no real-world fairness or demographic analysis is possible on this data.</li>
            </ul>
          </Glass>
        </div>

        <Glass index={4} className="p-6">
          <SectionHead title="Version history" tip="Every bundle the API has served or shadow-scored. Promotion swaps the model between two requests; an admin key is needed." />
          <Async query={models} rows={5}>
            {(rows) => (
              <ol className="flex flex-col gap-3">
                {rows.map((m) => (
                  <li key={m.version} className="glass flex flex-col gap-3 p-4" style={{ borderRadius: 18 }}>
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="mono break-all text-xs">{m.version}</span>
                      <span className="pill mono px-2.5 py-0.5 text-[0.65rem] uppercase tracking-widest" style={{ color: STAGE[m.stage], border: "1px solid currentColor" }}>
                        {m.stage}
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
                      <span className="faint">{m.promoted_at ? `promoted ${new Date(m.promoted_at).toLocaleString()}` : "in shadow"}</span>
                      {m.stage === "challenger" && (
                        <Tip tip="Serve this challenger from the next request on. The worker does this on its own when the challenger costs less on labelled traffic.">
                          <button
                            type="button"
                            className="btn btn-primary h-9 text-sm"
                            disabled={promote.isPending}
                            onClick={() =>
                              promote.mutate(m.version, {
                                onSuccess: (r) => toast(`Promoted ${r.version}`),
                                onError: (e) => toast(e instanceof Error ? e.message : "Promotion failed", "error"),
                              })
                            }
                          >
                            Promote
                          </button>
                        </Tip>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </Async>
        </Glass>
      </div>
    </>
  );
}
