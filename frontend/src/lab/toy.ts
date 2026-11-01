/** Seeded toy data with a real signal feature and a shortcut feature. */
import { mat } from "./nn";
import type { Env } from "./methods";

/** mulberry32: small, fast, seedable PRNG returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function normal(rand: () => number): () => number {
  return () => {
    const u = 1 - rand();
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

export const SIGNAL_NOISE = 1.5; // signal alone is ~75% accurate: Φ(1/1.5)
export const SHORTCUT_NOISE = 0.1;

/**
 * Features: [signal, shortcut]. signal = ±1 + N(0, 1.5) follows the true label.
 * shortcut = ±1 + N(0, 0.1) follows the label with probability `agree`.
 */
export function makeToyEnv(n: number, agree: number, rand: () => number): Env {
  const gauss = normal(rand);
  const X = mat(n, 2);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const label = rand() < 0.5 ? 1 : 0;
    const sign = 2 * label - 1;
    const sSign = rand() < agree ? sign : -sign;
    X.data[2 * i] = sign + SIGNAL_NOISE * gauss();
    X.data[2 * i + 1] = sSign + SHORTCUT_NOISE * gauss();
    y[i] = label;
  }
  return { X, y };
}

/** Same samples, with the shortcut column re-drawn to agree with probability `agree`.
 *  Lets the dial change test agreement without changing the signal noise. */
export function withShortcutAgreement(base: Env, flips: Float64Array, agree: number): Env {
  const X = mat(base.X.rows, 2, base.X.data);
  for (let i = 0; i < base.X.rows; i++) {
    const sign = 2 * base.y[i] - 1;
    const keep = flips[i] < agree;
    const s = base.X.data[2 * i + 1];
    const mag = Math.abs(s);
    X.data[2 * i + 1] = (keep ? sign : -sign) * mag;
  }
  return { X, y: base.y };
}
