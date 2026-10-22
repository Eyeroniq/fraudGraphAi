"use client";

import { createContext, type ReactNode, useCallback, useContext, useMemo, useRef, useState } from "react";

type Tone = "ok" | "error" | "info";
interface Toast {
  id: number;
  text: string;
  tone: Tone;
}

const Ctx = createContext<(text: string, tone?: Tone) => void>(() => {});

export function useToast() {
  return useContext(Ctx);
}

const COLOR: Record<Tone, string> = {
  ok: "var(--color-mint)",
  error: "var(--color-block)",
  info: "var(--color-stepup)",
};

export function Toaster({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);
  const push = useCallback((text: string, tone: Tone = "ok") => {
    const id = next.current++;
    setToasts((t) => [...t.slice(-3), { id, text, tone }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4_000);
  }, []);
  const value = useMemo(() => push, [push]);
  return (
    <Ctx.Provider value={value}>
      {children}
      <div aria-live="polite" className="pointer-events-none fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
        {toasts.map((t) => (
          <div key={t.id} role="status" className="glass pill toast-in flex items-center gap-3 px-5 py-3 text-sm">
            <span className="dot" style={{ background: COLOR[t.tone], boxShadow: `0 0 12px ${COLOR[t.tone]}` }} aria-hidden />
            {t.text}
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}
