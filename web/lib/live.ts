"use client";

// Live decision stream (`/ws/live`): one shared connection for the whole app, with
// reconnect + exponential backoff, and snapshots published at most every 250 ms so a
// busy stream never re-renders the page per message.

import { useSyncExternalStore } from "react";
import { type Action, liveUrl, MOCK, mockBackend } from "./api";

export interface LiveDecision {
  kind: "decision";
  txn_id: string;
  step: number;
  type: string;
  amount: number;
  decision: Action;
  p_fraud: number;
  model_version: string;
  at: number; // ms, when the browser received it
}

export type LiveStatus = "connecting" | "open" | "closed";

export interface LiveSnapshot {
  status: LiveStatus;
  recent: LiveDecision[]; // newest first
  counts: Record<Action, number>;
  buckets: { t: number; counts: Record<Action, number> }[]; // 5 s buckets, oldest first
  total: number;
}

const RECENT = 200;
const BUCKET_MS = 5_000;
const BUCKETS = 60; // five minutes
const PUBLISH_MS = 250;

const zero = (): Record<Action, number> => ({ APPROVE: 0, STEP_UP: 0, REVIEW: 0, BLOCK: 0 });

/** Delay before reconnect attempt `n` (0-based): 0.5 s doubling to 10 s, +-20 % jitter. */
export function backoff(attempt: number, random: () => number = Math.random): number {
  const base = Math.min(10_000, 500 * 2 ** attempt);
  return Math.round(base * (0.8 + 0.4 * random()));
}

/** Pure reducer: add one decision to a snapshot (exported for tests). */
export function addDecision(s: LiveSnapshot, d: LiveDecision): LiveSnapshot {
  const t = Math.floor(d.at / BUCKET_MS) * BUCKET_MS;
  const buckets = s.buckets.slice();
  const last = buckets[buckets.length - 1];
  if (last && last.t === t) {
    buckets[buckets.length - 1] = { t, counts: { ...last.counts, [d.decision]: last.counts[d.decision] + 1 } };
  } else {
    buckets.push({ t, counts: { ...zero(), [d.decision]: 1 } });
    while (buckets.length > BUCKETS) buckets.shift();
  }
  return {
    status: s.status,
    recent: [d, ...s.recent].slice(0, RECENT),
    counts: { ...s.counts, [d.decision]: s.counts[d.decision] + 1 },
    buckets,
    total: s.total + 1,
  };
}

export const emptySnapshot = (): LiveSnapshot => ({
  status: "connecting",
  recent: [],
  counts: zero(),
  buckets: [],
  total: 0,
});

class LiveStore {
  private snapshot = emptySnapshot();
  private pending = this.snapshot;
  private listeners = new Set<() => void>();
  private socket: WebSocket | null = null;
  private attempt = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private publishTimer: ReturnType<typeof setTimeout> | null = null;
  private started = false;
  private stopMock: (() => void) | null = null;

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    this.start();
    return () => {
      this.listeners.delete(listener);
    };
  };

  get = () => this.snapshot;

  private start() {
    if (this.started || typeof window === "undefined") return;
    this.started = true;
    if (MOCK) {
      void mockBackend().then((m) => {
        this.setStatus("open");
        this.stopMock = m.subscribeLive((d) => this.receive(d));
      });
      return;
    }
    this.connect();
  }

  private connect() {
    this.setStatus("connecting");
    let socket: WebSocket;
    try {
      socket = new WebSocket(liveUrl());
    } catch {
      this.retry();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      this.attempt = 0;
      this.setStatus("open");
    };
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string);
        if (data.kind === "decision") this.receive({ ...data, at: Date.now() });
      } catch {
        /* ignore malformed frames */
      }
    };
    socket.onclose = () => {
      this.socket = null;
      this.setStatus("closed");
      this.retry();
    };
  }

  private retry() {
    if (this.timer) return;
    const delay = backoff(this.attempt++);
    this.timer = setTimeout(() => {
      this.timer = null;
      this.connect();
    }, delay);
  }

  private receive(d: Omit<LiveDecision, "at"> & { at?: number }) {
    this.pending = addDecision(this.pending, { ...d, at: d.at ?? Date.now() } as LiveDecision);
    this.schedule();
  }

  private setStatus(status: LiveStatus) {
    this.pending = { ...this.pending, status };
    this.schedule(true);
  }

  private schedule(now = false) {
    if (now) {
      this.publish();
      return;
    }
    if (this.publishTimer) return;
    this.publishTimer = setTimeout(() => this.publish(), PUBLISH_MS);
  }

  private publish() {
    if (this.publishTimer) clearTimeout(this.publishTimer);
    this.publishTimer = null;
    this.snapshot = this.pending;
    for (const l of this.listeners) l();
  }
}

const store = new LiveStore();
const server = emptySnapshot();

export function useLiveFeed(): LiveSnapshot {
  return useSyncExternalStore(store.subscribe, store.get, () => server);
}
