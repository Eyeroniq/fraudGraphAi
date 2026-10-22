"use client";

// Hand-written SVG/HTML charts: a few kB instead of a charting library, so pages stay
// fast. Each chart has an accessible name; colours are never the only signal (legends
// and labels carry the meaning too).

import { type ReactNode, type RefObject, useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_W = 640;

/** Width of the wrapper in CSS pixels, so the SVG's 1 unit = 1 px and text stays small. */
function useWidth(): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(DEFAULT_W);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      const w = Math.round(entry.contentRect.width);
      if (w > 0) setWidth(w);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

function ticks(min: number, max: number, n = 4): number[] {
  if (!(max > min)) return [min];
  return Array.from({ length: n + 1 }, (_, i) => min + ((max - min) * i) / n);
}

export interface LineSeries {
  name: string;
  color: string;
  x: number[];
  y: (number | null)[];
  dashed?: boolean;
  width?: number;
}

/** Line chart with axes, legend, optional log x axis, hover read-out and markers. */
export function Lines({
  series,
  height = 260,
  xLog = false,
  xFmt = (v: number) => v.toFixed(2),
  yFmt = (v: number) => v.toFixed(2),
  yMin,
  yMax,
  xLabel,
  yLabel,
  markers = [],
  label,
}: {
  series: LineSeries[];
  height?: number;
  xLog?: boolean;
  xFmt?: (v: number) => string;
  yFmt?: (v: number) => string;
  yMin?: number;
  yMax?: number;
  xLabel?: string;
  yLabel?: string;
  markers?: { x: number; label: string; color?: string }[];
  label: string;
}) {
  const [box, W] = useWidth();
  const pad = { l: 56, r: 16, t: 14, b: xLabel ? 42 : 28 };
  const H = height;
  const fx = (v: number) => (xLog ? Math.log10(Math.max(v, 1e-12)) : v);
  const all = series.flatMap((s) => s.x.map(fx));
  const ys = series.flatMap((s) => s.y.filter((v): v is number => v !== null && Number.isFinite(v)));
  const x0 = Math.min(...all);
  const x1 = Math.max(...all);
  const y0 = yMin ?? Math.min(0, ...ys);
  const y1 = yMax ?? (Math.max(...ys) || 1);
  const sx = (v: number) => pad.l + ((fx(v) - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
  const sy = (v: number) => pad.t + (1 - (v - y0) / (y1 - y0 || 1)) * (H - pad.t - pad.b);
  const [hover, setHover] = useState<number | null>(null);

  const paths = useMemo(
    () =>
      series.map((s) => {
        let d = "";
        let pen = false;
        s.x.forEach((x, i) => {
          const y = s.y[i];
          if (y === null || !Number.isFinite(y)) {
            pen = false;
            return;
          }
          d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
          pen = true;
        });
        return d;
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [series, x0, x1, y0, y1, height, W],
  );

  const xTicks = xLog
    ? ticks(x0, x1, 4).map((v) => 10 ** v)
    : ticks(x0, x1, 4);
  const first = series[0];
  const hx = hover !== null && first ? first.x[hover] : null;

  return (
    <figure className="w-full" ref={box}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full overflow-visible"
        role="img"
        aria-label={label}
        onPointerMove={(e) => {
          if (!first) return;
          const r = e.currentTarget.getBoundingClientRect();
          const px = ((e.clientX - r.left) / r.width) * W;
          let best = 0;
          let dist = Infinity;
          first.x.forEach((x, i) => {
            const d = Math.abs(sx(x) - px);
            if (d < dist) {
              dist = d;
              best = i;
            }
          });
          setHover(best);
        }}
        onPointerLeave={() => setHover(null)}
      >
        {ticks(y0, y1, 4).map((v) => (
          <g key={`y${v}`}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(v)} y2={sy(v)} stroke="var(--grid)" />
            <text x={pad.l - 8} y={sy(v)} dy="0.32em" textAnchor="end" className="fill-[var(--text-faint)] text-[10px]">
              {yFmt(v)}
            </text>
          </g>
        ))}
        {xTicks.map((v) => (
          <text key={`x${v}`} x={sx(v)} y={H - pad.b + 16} textAnchor="middle" className="fill-[var(--text-faint)] text-[10px]">
            {xFmt(v)}
          </text>
        ))}
        {xLabel && (
          <text x={(W + pad.l) / 2} y={H - 6} textAnchor="middle" className="fill-[var(--text-muted)] text-[10px] uppercase tracking-[0.2em]">
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text transform={`translate(12 ${(H - pad.b) / 2}) rotate(-90)`} textAnchor="middle" className="fill-[var(--text-muted)] text-[10px] uppercase tracking-[0.2em]">
            {yLabel}
          </text>
        )}
        {markers.map((m) => (
          <g key={m.label}>
            <line x1={sx(m.x)} x2={sx(m.x)} y1={pad.t} y2={H - pad.b} stroke={m.color ?? "var(--text-faint)"} strokeDasharray="4 4" />
            <text x={sx(m.x) + 4} y={pad.t + 10} className="text-[10px]" fill={m.color ?? "var(--text-muted)"}>
              {m.label}
            </text>
          </g>
        ))}
        {paths.map((d, i) => (
          <path
            key={series[i].name}
            d={d}
            fill="none"
            stroke={series[i].color}
            strokeWidth={series[i].width ?? 2}
            strokeDasharray={series[i].dashed ? "5 5" : undefined}
            strokeLinejoin="round"
            strokeLinecap="round"
            className={series[i].dashed ? undefined : "draw"}
            style={{ "--len": 2400 } as React.CSSProperties}
          />
        ))}
        {hx !== null && hover !== null && (
          <g>
            <line x1={sx(hx)} x2={sx(hx)} y1={pad.t} y2={H - pad.b} stroke="var(--line-strong)" />
            {series.map((s) =>
              s.y[hover] === null || s.y[hover] === undefined ? null : (
                <circle key={s.name} cx={sx(s.x[hover])} cy={sy(s.y[hover] as number)} r={4} fill={s.color} stroke="var(--surface-solid)" strokeWidth={2} />
              ),
            )}
          </g>
        )}
      </svg>
      <figcaption className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
        {series.map((s) => (
          <span key={s.name} className="inline-flex items-center gap-2">
            <svg width="18" height="6" aria-hidden>
              <line x1="0" x2="18" y1="3" y2="3" stroke={s.color} strokeWidth="3" strokeDasharray={s.dashed ? "4 3" : undefined} />
            </svg>
            <span className="muted">{s.name}</span>
            {hover !== null && s.y[hover] !== null && s.y[hover] !== undefined && (
              <span className="num mono">{yFmt(s.y[hover] as number)}</span>
            )}
          </span>
        ))}
        {hx !== null && <span className="faint mono">@ {xFmt(hx)}</span>}
      </figcaption>
    </figure>
  );
}

/** Stacked areas over time (e.g. decision mix per 5 s). */
export function StackedArea({
  keys,
  colors,
  rows,
  height = 180,
  label,
}: {
  keys: string[];
  colors: Record<string, string>;
  rows: { t: number; values: Record<string, number> }[];
  height?: number;
  label: string;
}) {
  const [box, W] = useWidth();
  const H = height;
  const max = Math.max(1, ...rows.map((r) => keys.reduce((a, k) => a + (r.values[k] ?? 0), 0)));
  const n = Math.max(rows.length - 1, 1);
  const x = (i: number) => (i / n) * W;
  const y = (v: number) => H - (v / max) * (H - 8);
  let base = rows.map(() => 0);
  const layers = keys.map((k) => {
    const top = rows.map((r, i) => base[i] + (r.values[k] ?? 0));
    const d =
      rows.map((_, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(top[i]).toFixed(1)}`).join("") +
      rows
        .map((_, j) => {
          const i = rows.length - 1 - j;
          return `L${x(i).toFixed(1)},${y(base[i]).toFixed(1)}`;
        })
        .join("") +
      "Z";
    base = top;
    return { k, d };
  });
  return (
    <div ref={box} className="w-full">
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label} preserveAspectRatio="none">
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} x1={0} x2={W} y1={H * f} y2={H * f} stroke="var(--grid)" vectorEffect="non-scaling-stroke" />
      ))}
      {rows.length > 1 &&
        layers.map(({ k, d }) => (
          <path key={k} d={d} fill={colors[k]} fillOpacity={0.55} stroke={colors[k]} strokeWidth={1.2} vectorEffect="non-scaling-stroke" className="transition-[d] duration-700" />
        ))}
    </svg>
    </div>
  );
}

/** Histogram of values in [lo, hi]; log-y keeps the rare high scores visible. */
export function Histogram({
  values,
  bins = 24,
  lo = 0,
  hi = 1,
  color = "var(--color-mint)",
  height = 160,
  label,
  logY = true,
}: {
  values: number[];
  bins?: number;
  lo?: number;
  hi?: number;
  color?: string;
  height?: number;
  label: string;
  logY?: boolean;
}) {
  const counts = useMemo(() => {
    const c = Array(bins).fill(0) as number[];
    for (const v of values) c[Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)))]++;
    return c;
  }, [values, bins, lo, hi]);
  const scale = (v: number) => (logY ? Math.log1p(v) : v);
  const [box, W] = useWidth();
  const max = Math.max(1, ...counts.map(scale));
  const H = height;
  const bw = W / bins;
  return (
    <div ref={box} className="w-full">
    <svg viewBox={`0 0 ${W} ${H + 18}`} className="w-full" role="img" aria-label={label}>
      {counts.map((c, i) => {
        const h = (scale(c) / max) * (H - 6);
        return (
          <rect key={i} x={i * bw + 1.5} y={H - h} width={bw - 3} height={h} rx={3} fill={color} fillOpacity={0.35 + (0.6 * i) / bins} className="transition-all duration-700 ease-[var(--ease-spring)]">
            <title>{`${(lo + (i * (hi - lo)) / bins).toFixed(2)}–${(lo + ((i + 1) * (hi - lo)) / bins).toFixed(2)}: ${c}`}</title>
          </rect>
        );
      })}
      {[lo, (lo + hi) / 2, hi].map((v, i) => (
        <text key={v} x={i === 0 ? 0 : i === 1 ? W / 2 : W} y={H + 14} textAnchor={i === 0 ? "start" : i === 1 ? "middle" : "end"} className="fill-[var(--text-faint)] text-[10px]">
          {v.toFixed(1)}
        </text>
      ))}
    </svg>
    </div>
  );
}

/** Horizontal bars (HTML: crisp text, grows in with a spring). */
export function Bars({
  items,
  fmt = (v: number) => v.toFixed(2),
  max,
}: {
  items: { label: ReactNode; value: number; color?: string; note?: ReactNode }[];
  fmt?: (v: number) => string;
  max?: number;
}) {
  const m = max ?? Math.max(1e-12, ...items.map((i) => Math.abs(i.value)));
  return (
    <ul className="flex flex-col gap-3">
      {items.map((item, i) => (
        <li key={i} className="grid grid-cols-[minmax(6rem,11rem)_1fr_auto] items-center gap-3 text-sm">
          <span className="muted truncate">{item.label}</span>
          <span className="h-2.5 overflow-hidden rounded-full bg-[var(--grid)]">
            <span
              className="grow-x block h-full rounded-full"
              style={{ background: item.color ?? "var(--color-mint)", transform: `scaleX(${Math.abs(item.value) / m})` }}
            />
          </span>
          <span className="num mono text-xs">
            {fmt(item.value)}
            {item.note && <span className="faint ml-2">{item.note}</span>}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** SHAP waterfall: from the base margin, each contribution pushes the score up or down. */
export function Waterfall({ bias, items, label }: { bias: number; items: { label: string; value: number }[]; label: string }) {
  const steps = useMemo(() => {
    let run = bias;
    return items.map((it) => {
      const from = run;
      run += it.value;
      return { ...it, from, to: run };
    });
  }, [bias, items]);
  const all = [bias, ...steps.map((s) => s.to)];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pos = (v: number) => ((v - lo) / (hi - lo || 1)) * 100;
  const total = steps.length ? steps[steps.length - 1].to : bias;
  return (
    <div role="img" aria-label={label} className="flex flex-col gap-1.5 text-sm">
      <Row name="base margin" left={pos(Math.min(bias, bias))} width={0.6} color="var(--text-faint)" value={bias} />
      {steps.map((s) => (
        <Row
          key={s.label}
          name={s.label}
          left={pos(Math.min(s.from, s.to))}
          width={Math.max(0.6, Math.abs(pos(s.to) - pos(s.from)))}
          color={s.value >= 0 ? "var(--color-block)" : "var(--color-approve)"}
          value={s.value}
          signed
        />
      ))}
      <Row name="model margin" left={pos(total)} width={0.6} color="var(--text)" value={total} />
    </div>
  );
}

function Row({ name, left, width, color, value, signed = false }: { name: string; left: number; width: number; color: string; value: number; signed?: boolean }) {
  return (
    <div className="grid grid-cols-[minmax(7rem,12rem)_1fr_4.5rem] items-center gap-3">
      <span className="mono muted truncate text-xs">{name}</span>
      <span className="relative h-5">
        <span className="absolute inset-y-0 rounded-md transition-all duration-700 ease-[var(--ease-spring)]" style={{ left: `${left}%`, width: `${width}%`, background: color, opacity: 0.85 }} />
      </span>
      <span className="num mono text-right text-xs" style={{ color: signed ? color : undefined }}>
        {signed && value > 0 ? "+" : ""}
        {value.toFixed(3)}
      </span>
    </div>
  );
}

/** Feature x window heat map (HTML grid, values in each cell's title and aria label). */
export function Heatmap({
  rows,
  cols,
  value,
  max,
  alarm,
  colLabel = (c: number) => String(c),
  label,
}: {
  rows: string[];
  cols: number[];
  value: (row: string, col: number) => number | null;
  max: number;
  alarm?: (row: string, col: number) => boolean;
  colLabel?: (c: number) => string;
  label: string;
}) {
  return (
    <div role="table" aria-label={label} className="overflow-x-auto">
      <div className="grid gap-1" style={{ gridTemplateColumns: `minmax(9rem,max-content) repeat(${cols.length}, minmax(2.2rem,1fr))` }}>
        <div role="row" className="contents">
          <span role="columnheader">
            <span className="sr-only">Feature</span>
          </span>
          {cols.map((c) => (
            <span role="columnheader" key={c} className="mono faint text-center text-[10px]">
              {colLabel(c)}
            </span>
          ))}
        </div>
        {rows.map((r) => (
          <div role="row" key={r} className="contents">
            <span role="rowheader" className="mono muted truncate pr-2 text-xs leading-7">
              {r}
            </span>
            {cols.map((c) => {
              const v = value(r, c);
              const share = v === null ? 0 : Math.min(1, v / max);
              const hot = alarm?.(r, c);
              return (
                <span
                  role="cell"
                  key={c}
                  title={`${r} @ ${colLabel(c)}: PSI ${v === null ? "—" : v.toFixed(3)}${hot ? " (alarm)" : ""}`}
                  aria-label={`${r} ${colLabel(c)} ${v === null ? "no data" : v.toFixed(2)}${hot ? " alarm" : ""}`}
                  className="grid h-7 place-items-center rounded-md text-[10px] transition-colors duration-500"
                  style={{
                    background: v === null ? "var(--grid)" : `color-mix(in srgb, ${hot ? "var(--color-block)" : "var(--color-mint)"} ${Math.round(12 + share * 78)}%, transparent)`,
                    outline: hot ? "1px solid var(--color-block)" : undefined,
                  }}
                >
                  {hot ? "!" : ""}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
