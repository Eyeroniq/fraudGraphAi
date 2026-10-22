"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import { MOCK } from "@/lib/api";
import { useLiveFeed } from "@/lib/live";
import { ErrorBoundary } from "./ErrorBoundary";
import { Toaster } from "./Toaster";

const NAV = [
  { href: "/", label: "Live" },
  { href: "/cases", label: "Queue" },
  { href: "/lab", label: "Lab" },
  { href: "/drift", label: "Drift" },
  { href: "/explain", label: "Explain" },
  { href: "/model", label: "Model" },
];

function active(path: string, href: string) {
  return href === "/" ? path === "/" : path === href || path.startsWith(`${href}/`);
}

function Nav() {
  const path = usePathname();
  const refs = useRef<(HTMLAnchorElement | null)[]>([]);
  const [pill, setPill] = useState<{ left: number; width: number } | null>(null);
  const index = NAV.findIndex((n) => active(path, n.href));

  // The glass pill slides to the active link (measured, so it fits any label width).
  useLayoutEffect(() => {
    const el = refs.current[index];
    if (el) setPill({ left: el.offsetLeft, width: el.offsetWidth });
  }, [index]);

  return (
    <nav aria-label="Main" className="glass pill relative flex max-w-full overflow-x-auto p-1.5 [scrollbar-width:none]">
      {pill && (
        <span
          aria-hidden
          className="pill absolute top-1.5 bottom-1.5 bg-[rgb(255_255_255/0.1)] shadow-[inset_0_1px_0_rgb(255_255_255/0.15)] transition-[left,width] duration-500 ease-[var(--ease-spring)]"
          style={{ left: pill.left, width: pill.width }}
        />
      )}
      {NAV.map((n, i) => (
        <Link
          key={n.href}
          href={n.href}
          ref={(el) => {
            refs.current[i] = el;
          }}
          aria-current={i === index ? "page" : undefined}
          className={`pill relative px-4 py-2 text-[0.95rem] font-semibold transition-colors duration-300 sm:px-5 ${
            i === index ? "text-[var(--text)]" : "text-[var(--text-muted)] hover:text-[var(--text)]"
          }`}
        >
          {n.label}
        </Link>
      ))}
    </nav>
  );
}

function Status() {
  const live = useLiveFeed();
  const text = live.status === "open" ? (MOCK ? "demo data" : "stream live") : live.status === "connecting" ? "connecting" : "reconnecting";
  return (
    <span className="glass pill hidden items-center gap-3 px-5 py-2.5 md:inline-flex" role="status">
      <span
        className={`dot ${live.status === "open" ? "dot-live" : ""}`}
        style={live.status === "open" ? undefined : { background: "var(--color-review)", boxShadow: "none" }}
        aria-hidden
      />
      <span className="mono text-sm tracking-[0.12em]">{text}</span>
    </span>
  );
}

function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  useEffect(() => {
    setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  }, []);
  const flip = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      /* private mode */
    }
  };
  return (
    <button
      type="button"
      onClick={flip}
      className="glass pill grid h-11 w-11 place-items-center text-lg transition-transform duration-500 ease-[var(--ease-spring)] active:scale-90"
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
    >
      <span aria-hidden>{theme === "dark" ? "☾" : "☀"}</span>
    </button>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <Toaster>
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 glass pill px-4 py-2">
          Skip to content
        </a>
        <header className="sticky top-0 z-40 px-4 pt-4 sm:px-8">
          <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-4">
            <Link href="/" className="display flex items-center gap-1 text-xl tracking-tight" aria-label="FraudGraph home">
              FRAUDGRAPH<span className="mint-dot" aria-hidden />
            </Link>
            <Nav />
            <div className="flex items-center gap-3">
              <Status />
              <ThemeToggle />
            </div>
          </div>
        </header>
        <main id="main" className="mx-auto max-w-[1400px] px-4 pb-24 pt-12 sm:px-8 sm:pt-16">
          <ErrorBoundary>{children}</ErrorBoundary>
        </main>
      </Toaster>
    </QueryClientProvider>
  );
}
