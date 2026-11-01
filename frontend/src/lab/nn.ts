/**
 * TypeScript port of src/irm_stock/core/{mlp,losses,optim}.py on Float64Array.
 * Matrices are row-major. Keep this file in step with the Python core:
 * parity is enforced by nn.test.ts against fixtures from scripts/export_lab_fixtures.py.
 *
 * Shapes: X (n×d), W (h×d), V (h) [Python (1×h)], Z/H (n×h), logits F (n).
 */

export interface Mat {
  rows: number;
  cols: number;
  data: Float64Array;
}

export const mat = (rows: number, cols: number, data?: ArrayLike<number>): Mat => ({
  rows,
  cols,
  data: data ? Float64Array.from(data) : new Float64Array(rows * cols),
});

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export interface Forward {
  Z: Mat;
  H: Mat;
  F: Float64Array;
}

export function forward(X: Mat, W: Mat, V: Float64Array): Forward {
  const n = X.rows;
  const d = X.cols;
  const h = W.rows;
  const Z = mat(n, h);
  const H = mat(n, h);
  const F = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let f = 0;
    for (let j = 0; j < h; j++) {
      let z = 0;
      for (let k = 0; k < d; k++) z += X.data[i * d + k] * W.data[j * d + k];
      Z.data[i * h + j] = z;
      const a = z > 0 ? z : 0;
      H.data[i * h + j] = a;
      f += a * V[j];
    }
    F[i] = f;
  }
  return { Z, H, F };
}

const LOG_EPS = 1e-7;

export function meanNll(logits: Float64Array, y: Float64Array): number {
  let s = 0;
  for (let i = 0; i < logits.length; i++) {
    const p = sigmoid(logits[i]);
    s += y[i] * Math.log(p + LOG_EPS) + (1 - y[i]) * Math.log(1 - p + LOG_EPS);
  }
  return -s / logits.length;
}

export function meanAccuracy(logits: Float64Array, y: Float64Array): number {
  let c = 0;
  for (let i = 0; i < logits.length; i++) if ((logits[i] > 0 ? 1 : 0) === y[i]) c++;
  return c / logits.length;
}

export function irmPenalty(logits: Float64Array, y: Float64Array): number {
  let g = 0;
  for (let i = 0; i < logits.length; i++) g += logits[i] * (sigmoid(logits[i]) - y[i]);
  g /= logits.length;
  return g * g;
}

export interface Grads {
  dW: Mat;
  dV: Float64Array;
}

/** Backprop a per-sample gradient on the logits (already divided by n if needed). */
function backprop(X: Mat, Z: Mat, H: Mat, V: Float64Array, dF: Float64Array, scale: number): Grads {
  const n = X.rows;
  const d = X.cols;
  const h = V.length;
  const dW = mat(h, d);
  const dV = new Float64Array(h);
  for (let i = 0; i < n; i++) {
    const g = dF[i];
    if (g === 0) continue;
    for (let j = 0; j < h; j++) {
      dV[j] += g * H.data[i * h + j];
      if (Z.data[i * h + j] > 0) {
        const gz = g * V[j];
        for (let k = 0; k < d; k++) dW.data[j * d + k] += gz * X.data[i * d + k];
      }
    }
  }
  for (let j = 0; j < h; j++) dV[j] *= scale;
  for (let q = 0; q < dW.data.length; q++) dW.data[q] *= scale;
  return { dW, dV };
}

export function nllGradients(X: Mat, y: Float64Array, fw: Forward, V: Float64Array): Grads {
  const dF = fw.F.map((f, i) => sigmoid(f) - y[i]);
  return backprop(X, fw.Z, fw.H, V, dF, 1 / X.rows);
}

export function irmPenaltyGradients(X: Mat, y: Float64Array, fw: Forward, V: Float64Array): Grads {
  const n = X.rows;
  let gs = 0;
  for (let i = 0; i < n; i++) gs += fw.F[i] * (sigmoid(fw.F[i]) - y[i]);
  gs /= n;
  const dF = fw.F.map((f, i) => {
    const s = sigmoid(f);
    return 2 * gs * (((s - y[i]) + f * s * (1 - s)) / n);
  });
  return backprop(X, fw.Z, fw.H, V, dF, 1);
}

/** Gradient of the ensemble NLL w.r.t. one member, as in ensemble_member_nll_gradients. */
export function ensembleMemberNllGradients(
  X: Mat,
  y: Float64Array,
  fw: Forward,
  ensLogits: Float64Array,
  V: Float64Array,
  numModels: number,
): Grads {
  const dF = ensLogits.map((g, i) => (1 / numModels) * (sigmoid(g) - y[i]));
  return backprop(X, fw.Z, fw.H, V, dF, 1 / X.rows);
}

export function ensembleLogits(X: Mat, Ws: Mat[], Vs: Float64Array[]): Float64Array {
  const out = new Float64Array(X.rows);
  for (let e = 0; e < Ws.length; e++) {
    const { F } = forward(X, Ws[e], Vs[e]);
    for (let i = 0; i < out.length; i++) out[i] += F[i];
  }
  return out.map((v) => v / Ws.length);
}

/** Glorot-uniform init; W is drawn before V, like init_mlp. */
export function initMlp(d: number, h: number, uniform: () => number): { W: Mat; V: Float64Array } {
  const lw = Math.sqrt(6 / (d + h));
  const W = mat(h, d);
  for (let q = 0; q < W.data.length; q++) W.data[q] = -lw + 2 * lw * uniform();
  const lv = Math.sqrt(6 / (h + 1));
  const V = new Float64Array(h).map(() => -lv + 2 * lv * uniform());
  return { W, V };
}

export class Adam {
  private m: Float64Array[];
  private v: Float64Array[];
  constructor(
    private params: Float64Array[],
    public lr: number,
    private b1 = 0.9,
    private b2 = 0.999,
    private eps = 1e-8,
  ) {
    this.m = params.map((p) => new Float64Array(p.length));
    this.v = params.map((p) => new Float64Array(p.length));
  }

  /** In-place update; t is the 1-based bias-correction step. */
  step(grads: Float64Array[], t: number) {
    const { b1, b2 } = this;
    const c1 = 1 - b1 ** t;
    const c2 = 1 - b2 ** t;
    this.params.forEach((p, k) => {
      const g = grads[k];
      const m = this.m[k];
      const v = this.v[k];
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        p[i] -= (this.lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + this.eps);
      }
    });
  }
}
