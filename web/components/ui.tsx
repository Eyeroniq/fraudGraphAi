"use client";

import {
  type CSSProperties,
  type HTMLAttributes,
  type ReactNode,
  useCallback,
  useRef,
} from "react";
import type { Action } from "@/lib/api";
import { ACTION_META, setLabel } from "@/lib/format";

/** Pointer-tracked specular highlight for glass surfaces (rAF-throttled, passive). */
export function useSpecular<T extends HTMLElement>() {
  const frame = useRef(0);
  const onPointerMove = useCallback((e: React.PointerEvent<T>) => {
    const el = e.currentTarget;
    const { clientX, clientY } = e;
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => {
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${clientX - r.left}px`);
      el.style.setProperty("--my", `${clientY - r.top}px`);
      el.style.setProperty("--spec", "1");
    });
  }, []);
  const onPointerLeave = useCallback((e: React.PointerEvent<T>) => {
    e.currentTarget.style.setProperty("--spec", "0.7");
  }, []);
  return { onPointerMove, onPointerLeave };
}

export function Glass({
  children,
  className = "",
  hover = false,
  index,
  style,
  as: Tag = "section",
  ...rest
}: HTMLAttributes<HTMLElement> & { hover?: boolean; index?: number; as?: "section" | "div" | "article" }) {
  const spec = useSpecular<HTMLElement>();
  const s = index === undefined ? style : ({ ...style, "--i": index } as CSSProperties);
  return (
    <Tag className={`glass ${hover ? "glass-hover" : ""} rise ${className}`} style={s} {...spec} {...rest}>
      {children}
    </Tag>
  );
}

/** Mono, letter-spaced label in a dark pill with a mint dot ("● HI, I'M"). */
export function Eyebrow({ children, live = false }: { children: ReactNode; live?: boolean }) {
  return (
    <span className="glass pill inline-flex items-center gap-3 px-5 py-2.5">
      <span className={`dot ${live ? "dot-live" : ""}`} aria-hidden />
      <span className="label-mono" style={{ color: "var(--text)" }}>
        {children}
      </span>
    </span>
  );
}

/** Display title with the mint full stop, and a ruled mono subtitle. */
export function PageTitle({ title, subtitle, children }: { title: string; subtitle?: string; children?: ReactNode }) {
  return (
    <header className="rise mb-10 flex flex-col gap-6" style={{ "--i": 0 } as CSSProperties}>
      <h1 className="display text-[clamp(2.8rem,7vw,5.6rem)]">
        {title}
        <span className="mint-dot" aria-hidden />
      </h1>
      {subtitle && (
        <p className="flex items-center gap-5">
          <span aria-hidden className="h-px w-12 bg-[var(--color-mint)]" />
          <span className="label-mono" style={{ color: "var(--text)" }}>
            {subtitle}
          </span>
        </p>
      )}
      {children}
    </header>
  );
}

export function Tip({ children, tip }: { children: ReactNode; tip: string }) {
  return (
    <span className="tip" tabIndex={0}>
      {children}
      <span aria-hidden className="faint text-[0.7rem]">
        ⓘ
      </span>
      <span role="tooltip" className="tip-bubble">
        {tip}
      </span>
    </span>
  );
}

export function Kpi({
  label,
  value,
  tip,
  sub,
  index,
  accent,
}: {
  label: string;
  value: ReactNode;
  tip: string;
  sub?: ReactNode;
  index?: number;
  accent?: string;
}) {
  return (
    <Glass hover index={index} className="flex min-h-[8.5rem] flex-col justify-between p-5" as="div">
      <Tip tip={tip}>
        <span className="label-mono text-[0.62rem]">{label}</span>
      </Tip>
      <div className="num text-3xl font-semibold tracking-tight" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      {sub && <div className="faint text-xs">{sub}</div>}
    </Glass>
  );
}

export function SectionHead({ title, tip, right }: { title: string; tip?: string; right?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
      <h2 className="label-mono" style={{ color: "var(--text)" }}>
        {tip ? <Tip tip={tip}>{title}</Tip> : title}
      </h2>
      {right}
    </div>
  );
}

export function DecisionBadge({ decision, size = "sm" }: { decision: Action | string | null | undefined; size?: "sm" | "lg" }) {
  if (!decision) return <span className="faint">—</span>;
  const meta = ACTION_META[decision as Action];
  if (!meta) return <span>{decision}</span>;
  const big = size === "lg";
  return (
    <span
      className={`pill inline-flex items-center gap-2 font-semibold ${big ? "px-5 py-2 text-lg" : "px-3 py-1 text-xs"}`}
      style={{ color: meta.color, background: `color-mix(in srgb, ${meta.color} 13%, transparent)`, border: `1px solid color-mix(in srgb, ${meta.color} 40%, transparent)` }}
      title={meta.help}
    >
      <span aria-hidden>{meta.icon}</span>
      {meta.label}
    </span>
  );
}

export function ConformalChip({ set }: { set: string[] | null | undefined }) {
  const label = setLabel(set);
  if (!label) return null;
  const uncertain = label === "uncertain" || label === "no fit";
  return (
    <Tip
      tip={
        uncertain
          ? "Conformal prediction could not rule out either answer at the chosen error rate: the model is not sure."
          : `Conformal set {${label}}: at the chosen error rate only this answer fits.`
      }
    >
      <span
        className="pill mono inline-flex items-center gap-2 px-3 py-1 text-xs uppercase tracking-widest"
        style={{
          color: uncertain ? "var(--color-review)" : "var(--text-muted)",
          border: `1px ${uncertain ? "dashed" : "solid"} currentColor`,
        }}
      >
        {uncertain ? "◐" : "●"} {label}
      </span>
    </Tip>
  );
}

export function Skeleton({ className = "h-24" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

export function Loading({ rows = 3 }: { rows?: number }) {
  return (
    <div className="flex flex-col gap-3" role="status" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-10" />
      ))}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-3 py-12 text-center">
      <span className="dot" aria-hidden />
      <p className="font-semibold">{title}</p>
      {children && <div className="muted max-w-md text-sm">{children}</div>}
    </div>
  );
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="flex flex-col items-center gap-3 py-10 text-center">
      <span className="text-[var(--color-block)]" aria-hidden>
        ✕
      </span>
      <p className="font-semibold">Could not load this</p>
      <p className="muted max-w-md text-sm">{message}</p>
      {retry && (
        <button type="button" className="btn btn-primary h-10" onClick={retry}>
          Try again
        </button>
      )}
    </div>
  );
}

/** Loading / error / empty / content, in one place. */
export function Async<T>({
  query,
  empty,
  rows,
  children,
}: {
  query: { data: T | undefined; error: unknown; isLoading: boolean; refetch: () => unknown };
  empty?: (data: T) => ReactNode | null;
  rows?: number;
  children: (data: T) => ReactNode;
}) {
  if (query.isLoading) return <Loading rows={rows} />;
  if (query.error) return <ErrorState error={query.error} retry={() => void query.refetch()} />;
  if (query.data === undefined) return <Loading rows={rows} />;
  const e = empty?.(query.data);
  return <>{e ?? children(query.data)}</>;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="glass pill inline-flex gap-1 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className="pill px-4 py-1.5 text-sm transition-colors duration-300"
          style={
            value === o.value
              ? { background: "rgb(110 231 183 / 0.16)", color: "var(--color-mint)" }
              : { color: "var(--text-muted)" }
          }
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Meter({ value, max, color = "var(--color-mint)", label }: { value: number; max: number; color?: string; label: string }) {
  const share = max > 0 ? Math.min(1, value / max) : 0;
  return (
    <div role="meter" aria-label={label} aria-valuenow={value} aria-valuemin={0} aria-valuemax={max} className="h-2 w-full overflow-hidden rounded-full bg-[var(--grid)]">
      <div className="grow-x h-full rounded-full" style={{ background: color, transform: `scaleX(${share})` }} />
    </div>
  );
}
