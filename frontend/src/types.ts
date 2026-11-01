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
  /** Display name, e.g. "S&P 500 (SPY)". */
  label?: string;
  /** Set for runs computed with the current package (not the original report logs). */
  computed_on?: string;
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

export type Variant = "shortcut_flipped" | "no_shortcut";
export type Accs = Record<MethodKey, number>;

export interface LiveTicker {
  n_days: number;
  as_of: string;
  base_rate: number;
  variants: Record<Variant, Accs>;
  monthly: ({ month: string; n_days: number; base_rate: number } & Record<Variant, Accs>)[];
  regime: { trend: "bull" | "bear" | "side" | "unknown"; volatility: "low" | "mid" | "high"; vol_20d: number };
}

export interface LiveSnapshot {
  week: string;
  run_date: string;
  train: { start: string; end: string; env_mode: string };
  tickers: Record<string, LiveTicker>;
  errors: Record<string, string>;
}
