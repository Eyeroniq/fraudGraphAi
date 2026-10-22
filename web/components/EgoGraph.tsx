"use client";

// 2-hop ego-graph of a transaction's accounts. A small deterministic force layout
// (Fruchterman-Reingold, ~60 nodes) replaces a graph library: no extra download.

import { useMemo, useState } from "react";
import type { GraphResponse } from "@/lib/api";
import { money } from "@/lib/format";

const W = 640;
const H = 420;

interface P {
  x: number;
  y: number;
}

function layout(nodes: GraphResponse["nodes"], edges: GraphResponse["edges"]): Map<string, P> {
  const n = nodes.length;
  const pos = new Map<string, P>();
  let seed = 7;
  const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  nodes.forEach((node, i) => {
    const a = (i / Math.max(n, 1)) * Math.PI * 2;
    const r = node.center ? 20 : 120 + rand() * 60;
    pos.set(node.id, { x: W / 2 + r * Math.cos(a), y: H / 2 + r * Math.sin(a) });
  });
  const k = Math.sqrt((W * H) / Math.max(n, 1)) * 0.55;
  let temp = 60;
  for (let it = 0; it < 160; it++) {
    const disp = new Map<string, P>(nodes.map((v) => [v.id, { x: 0, y: 0 }]));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = pos.get(nodes[i].id)!;
        const b = pos.get(nodes[j].id)!;
        const dx = a.x - b.x || 0.01;
        const dy = a.y - b.y || 0.01;
        const d = Math.hypot(dx, dy);
        const f = (k * k) / d;
        const da = disp.get(nodes[i].id)!;
        const db = disp.get(nodes[j].id)!;
        da.x += (dx / d) * f;
        da.y += (dy / d) * f;
        db.x -= (dx / d) * f;
        db.y -= (dy / d) * f;
      }
    }
    for (const e of edges) {
      const a = pos.get(e.source);
      const b = pos.get(e.target);
      if (!a || !b || e.source === e.target) continue;
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const d = Math.hypot(dx, dy) || 0.01;
      const f = (d * d) / k;
      disp.get(e.source)!.x -= (dx / d) * f;
      disp.get(e.source)!.y -= (dy / d) * f;
      disp.get(e.target)!.x += (dx / d) * f;
      disp.get(e.target)!.y += (dy / d) * f;
    }
    for (const node of nodes) {
      const p = pos.get(node.id)!;
      const d = disp.get(node.id)!;
      const len = Math.hypot(d.x, d.y) || 1;
      p.x += (d.x / len) * Math.min(len, temp);
      p.y += (d.y / len) * Math.min(len, temp);
      // gentle pull to the middle keeps disconnected pieces on screen
      p.x += (W / 2 - p.x) * 0.005;
      p.y += (H / 2 - p.y) * 0.005;
      p.x = Math.min(W - 24, Math.max(24, p.x));
      p.y = Math.min(H - 24, Math.max(24, p.y));
    }
    temp *= 0.96;
  }
  // Fit to the box: a small graph should fill it, not huddle in the middle.
  const xs = [...pos.values()].map((p) => p.x);
  const ys = [...pos.values()].map((p) => p.y);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const scale = Math.min((W - 80) / (maxX - minX || 1), (H - 80) / (maxY - minY || 1), 3);
  for (const p of pos.values()) {
    p.x = W / 2 + (p.x - (minX + maxX) / 2) * scale;
    p.y = H / 2 + (p.y - (minY + maxY) / 2) * scale;
  }
  return pos;
}

export function EgoGraph({ graph, focusTxn }: { graph: GraphResponse; focusTxn: string }) {
  const pos = useMemo(() => layout(graph.nodes, graph.edges), [graph]);
  const degree = useMemo(() => {
    const d = new Map<string, number>();
    for (const e of graph.edges) {
      d.set(e.source, (d.get(e.source) ?? 0) + 1);
      d.set(e.target, (d.get(e.target) ?? 0) + 1);
    }
    return d;
  }, [graph]);
  const [selected, setSelected] = useState<string | null>(null);
  const timeline = selected
    ? graph.edges.filter((e) => e.source === selected || e.target === selected).sort((a, b) => b.step - a.step)
    : [];

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_16rem]">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="group" aria-label={`Payment graph around ${focusTxn}: ${graph.nodes.length} accounts, ${graph.edges.length} payments`}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--text-faint)" />
          </marker>
        </defs>
        {graph.edges.map((e) => {
          const a = pos.get(e.source);
          const b = pos.get(e.target);
          if (!a || !b) return null;
          const focus = e.txn_id === focusTxn;
          const color = e.is_fraud ? "var(--color-block)" : e.is_fraud === false ? "var(--color-approve)" : "var(--text-faint)";
          return (
            <line
              key={e.txn_id}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={focus ? "var(--color-review)" : color}
              strokeWidth={focus ? 3 : 1.3}
              strokeOpacity={focus ? 1 : 0.7}
              markerEnd="url(#arrow)"
            >
              <title>{`${e.type} ${money(e.amount)} at step ${e.step}${e.is_fraud ? " (fraud)" : ""}`}</title>
            </line>
          );
        })}
        {graph.nodes.map((node) => {
          const p = pos.get(node.id)!;
          const r = node.center ? 11 : 7;
          const fresh = (degree.get(node.id) ?? 0) <= 1;
          const active = selected === node.id;
          return (
            <g
              key={node.id}
              transform={`translate(${p.x} ${p.y})`}
              tabIndex={0}
              role="button"
              aria-label={`${node.kind} ${node.id}${node.center ? " (in this payment)" : ""}`}
              onClick={() => setSelected(node.id)}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setSelected(node.id)}
              className="cursor-pointer outline-none"
            >
              {node.kind === "merchant" ? (
                <rect x={-r} y={-r} width={r * 2} height={r * 2} rx={3} fill="var(--surface-solid)" stroke="var(--color-stepup)" strokeWidth={active ? 3 : 1.5} />
              ) : (
                <circle
                  r={r}
                  fill={node.center ? "var(--color-mint)" : "var(--surface-solid)"}
                  stroke={node.center ? "var(--color-mint)" : "var(--text-muted)"}
                  strokeWidth={active ? 3 : 1.5}
                  strokeDasharray={fresh && !node.center ? "3 2" : undefined}
                />
              )}
              {(node.center || active) && (
                <text y={-r - 6} textAnchor="middle" className="mono fill-[var(--text)] text-[10px]">
                  {node.id}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <aside className="text-sm" aria-live="polite">
        <p className="label-mono mb-3 text-[0.62rem]">Legend</p>
        <ul className="muted mb-5 flex flex-col gap-1.5 text-xs">
          <li><span className="mr-2 inline-block h-2 w-4 rounded bg-[var(--color-block)]" />fraud (labelled)</li>
          <li><span className="mr-2 inline-block h-2 w-4 rounded bg-[var(--color-approve)]" />legit (labelled)</li>
          <li><span className="mr-2 inline-block h-2 w-4 rounded bg-[var(--color-review)]" />this payment</li>
          <li><span className="mr-2 inline-block h-3 w-3 rounded-full border border-dashed border-[var(--text-muted)]" />seen once (new)</li>
          <li><span className="mr-2 inline-block h-3 w-3 rounded-sm border border-[var(--color-stepup)]" />merchant</li>
        </ul>
        {selected ? (
          <>
            <p className="label-mono mb-2 text-[0.62rem]">{selected}</p>
            <ol className="flex max-h-56 flex-col gap-1.5 overflow-y-auto">
              {timeline.map((e) => (
                <li key={e.txn_id} className="mono flex justify-between gap-2 text-xs">
                  <span className="faint">s{e.step}</span>
                  <span className={e.source === selected ? "" : "text-[var(--color-mint)]"}>{e.source === selected ? "→ out" : "← in"}</span>
                  <span className="num">{money(e.amount)}</span>
                  {e.is_fraud && <span className="text-[var(--color-block)]">fraud</span>}
                </li>
              ))}
            </ol>
          </>
        ) : (
          <p className="faint text-xs">Select an account to see its payments.</p>
        )}
      </aside>
    </div>
  );
}
