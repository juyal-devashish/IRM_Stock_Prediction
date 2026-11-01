export type MethodKey = "erm" | "irm" | "irm_games";

export interface MethodStats {
  train_mean: number;
  train_std: number;
  test_mean: number;
  test_std: number;
  n_restarts?: number;
}

export interface Run {
  id: string;
  ticker: string | null;
  source: string;
  methods: Record<MethodKey, MethodStats>;
}

export interface Benchmark {
  id: string;
  label: string;
  description: string;
  runs: Run[];
}

export interface Figure {
  file: string;
  caption: string;
}

export interface Results {
  benchmarks: Benchmark[];
  figures: Figure[];
}
