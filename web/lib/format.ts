import type { Action } from "./api";

const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });
const whole = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });

/** 1234567 -> "1.2M"; small values stay exact. */
export function money(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Math.abs(value) < 10_000 ? whole.format(value) : compact.format(value);
}

export function count(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Math.abs(value) < 10_000 ? whole.format(value) : compact.format(value);
}

/** 0.0123 -> "1.2%" (digits after the point adapt to tiny shares). */
export function pct(value: number | null | undefined, digits?: number): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const d = digits ?? (Math.abs(value) < 0.01 && value !== 0 ? 2 : 1);
  return `${(value * 100).toFixed(d)}%`;
}

export function prob(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value < 0.001 && value > 0 ? value.toExponential(1) : value.toFixed(3);
}

export function ms(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;
}

/** Age of an ISO timestamp, e.g. "3m", "2h 05m". */
export function age(iso: string, now = Date.now()): string {
  const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${String(m % 60).padStart(2, "0")}m`;
  return `${Math.floor(h / 24)}d`;
}

export const ACTIONS: Action[] = ["APPROVE", "STEP_UP", "REVIEW", "BLOCK"];

export const ACTION_META: Record<Action, { label: string; icon: string; color: string; help: string }> = {
  APPROVE: { label: "Approve", icon: "✓", color: "var(--color-approve)", help: "Let the payment through." },
  STEP_UP: {
    label: "Step-up",
    icon: "⇡",
    color: "var(--color-stepup)",
    help: "Challenge the customer (OTP / 3DS) before authorising.",
  },
  REVIEW: { label: "Review", icon: "⚑", color: "var(--color-review)", help: "Hold the payment for an analyst." },
  BLOCK: { label: "Block", icon: "✕", color: "var(--color-block)", help: "Decline the payment." },
};

export const REASON_TEXT: Record<string, string> = {
  expected_cost: "cheapest expected cost",
  conformal_fraud: "conformal set says fraud: at least a review",
  model_uncertain: "model uncertain: both answers plausible",
  capacity_overflow: "review team at capacity today",
  type_not_modelled: "payment type not modelled",
};

/** Conformal set -> what it means for a reader. */
export function setLabel(set: string[] | null | undefined): string | null {
  if (!set) return null;
  if (set.length === 2) return "uncertain";
  if (set.length === 0) return "no fit";
  return set[0];
}
