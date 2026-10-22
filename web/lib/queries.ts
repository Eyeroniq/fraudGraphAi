"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Scenario } from "./api";

// Monitoring data refreshes on its own; report data is fetched once per session.
const LIVE = { refetchInterval: 5_000, staleTime: 2_000 };

export const useCases = (p: { status?: string; sort?: string; limit?: number; offset?: number }) =>
  useQuery({ queryKey: ["cases", p], queryFn: () => api.cases(p), placeholderData: keepPreviousData, ...LIVE });

export const useCase = (id: number | null) =>
  useQuery({ queryKey: ["case", id], queryFn: () => api.case(id as number), enabled: id !== null });

export const useTransaction = (txn: string | undefined) =>
  useQuery({ queryKey: ["txn", txn], queryFn: () => api.transaction(txn as string), enabled: !!txn });

export const useExplain = (txn: string | undefined) =>
  useQuery({ queryKey: ["explain", txn], queryFn: () => api.explain(txn as string), enabled: !!txn });

export const useGraph = (txn: string | undefined) =>
  useQuery({ queryKey: ["graph", txn], queryFn: () => api.graph(txn as string), enabled: !!txn });

export const useDrift = (window = "1w") =>
  useQuery({ queryKey: ["drift", window], queryFn: () => api.drift(window), ...LIVE });

export const usePerformance = (window = "1w") =>
  useQuery({ queryKey: ["performance", window], queryFn: () => api.performance(window), ...LIVE });

export const useModels = () => useQuery({ queryKey: ["models"], queryFn: api.models, ...LIVE });

export const useCurrentModel = () =>
  useQuery({ queryKey: ["current-model"], queryFn: api.currentModel, staleTime: 30_000 });

export const useLab = () => useQuery({ queryKey: ["lab"], queryFn: api.lab, staleTime: Infinity });

export const useSim = () => useQuery({ queryKey: ["sim"], queryFn: api.simStatus, refetchInterval: 2_000 });

export function useResolve() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ id, verdict }: { id: number; verdict: "fraud" | "legit" }) => api.resolve(id, verdict),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["cases"] });
      void client.invalidateQueries({ queryKey: ["case"] });
      void client.invalidateQueries({ queryKey: ["performance"] });
    },
  });
}

export function usePromote() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (version: string) => api.promote(version),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["current-model"] });
    },
  });
}

export function useSimControl() {
  const client = useQueryClient();
  const done = () => void client.invalidateQueries({ queryKey: ["sim"] });
  return {
    start: useMutation({ mutationFn: api.simStart, onSuccess: done }),
    stop: useMutation({ mutationFn: api.simStop, onSuccess: done }),
    scenario: useMutation({ mutationFn: (name: Scenario) => api.simScenario(name), onSuccess: done }),
  };
}
