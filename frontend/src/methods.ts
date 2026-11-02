import type { MethodKey } from "./types";

export const METHOD_ORDER: MethodKey[] = ["erm", "irm", "irm_games"];

/** Methods drawn in charts. IRM-Games appears in tables only. */
export const CHART_METHODS: MethodKey[] = ["erm", "irm"];

export const METHOD_INFO: Record<MethodKey, { label: string; blurb: string }> = {
  erm: {
    label: "ERM",
    blurb: "Standard training: minimise average error on everything it has seen.",
  },
  irm: {
    label: "IRM",
    blurb: "Adds a penalty on features whose usefulness changes between environments.",
  },
  irm_games: {
    label: "IRM-Games",
    blurb: "One model per environment, trained against the others' average prediction.",
  },
};

/** Share of samples where the shortcut feature agrees with the label, per environment.
 *  Derived from the environment builders (agreement = 1 - e). */
export const SHORTCUT: Record<string, { env: string; agree: number; held_out: boolean }[]> = {
  "colored-mnist": [
    { env: "Training 1", agree: 0.8, held_out: false },
    { env: "Training 2", agree: 0.9, held_out: false },
    { env: "Test", agree: 0.1, held_out: true },
  ],
  "time-split": [
    { env: "Early years", agree: 0.8, held_out: false },
    { env: "Middle years", agree: 0.9, held_out: false },
    { env: "Recent years", agree: 0.1, held_out: true },
  ],
  "market-regimes": [
    { env: "Bull market", agree: 0.8, held_out: false },
    { env: "Sideways market", agree: 0.8, held_out: false },
    { env: "Bear market", agree: 0.1, held_out: true },
  ],
  volatility: [
    { env: "Low volatility", agree: 0.8, held_out: false },
    { env: "Mid volatility", agree: 0.9, held_out: false },
    { env: "High volatility", agree: 0.1, held_out: true },
  ],
};
