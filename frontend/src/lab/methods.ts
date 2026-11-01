/**
 * Incremental ERM / IRMv1 / IRM-Games trainers mirroring
 * src/irm_stock/methods/mlp_methods.py. One call to `step()` is one optimiser step
 * (for IRM-Games: one player's turn; a full Python "step" is one round of turns).
 */
import {
  Adam,
  ensembleLogits,
  ensembleMemberNllGradients,
  forward,
  initMlp,
  irmPenaltyGradients,
  type Mat,
  mat,
  nllGradients,
} from "./nn";

export interface Env {
  X: Mat;
  y: Float64Array;
}

export interface BaseConfig {
  hidden: number;
  lr: number;
  l2: number;
}

/** Draws `size` row indices in [0, n) with replacement (numpy's randint). */
export type BatchSampler = (n: number, size: number) => Int32Array;

export interface Trainer {
  readonly kind: "erm" | "irm" | "irm_games";
  /** Optimiser steps taken so far (IRM-Games: player turns). */
  readonly steps: number;
  step(): void;
  logits(X: Mat): Float64Array;
  /** Per-feature input gradient magnitudes, averaged over X (for "what does it rely on"). */
  reliance(X: Mat): Float64Array;
}

function addScaled(a: Float64Array, b: Float64Array, s: number) {
  for (let i = 0; i < a.length; i++) a[i] += s * b[i];
}

function inputSensitivity(X: Mat, Ws: Mat[], Vs: Float64Array[]): Float64Array {
  const d = X.cols;
  const out = new Float64Array(d);
  for (let e = 0; e < Ws.length; e++) {
    const W = Ws[e];
    const V = Vs[e];
    const { Z } = forward(X, W, V);
    const h = W.rows;
    for (let i = 0; i < X.rows; i++) {
      for (let k = 0; k < d; k++) {
        let g = 0;
        for (let j = 0; j < h; j++) if (Z.data[i * h + j] > 0) g += V[j] * W.data[j * d + k];
        out[k] += Math.abs(g / Ws.length);
      }
    }
  }
  return out.map((v) => v / X.rows);
}

function pool(envs: Env[]): Env {
  const d = envs[0].X.cols;
  const n = envs.reduce((s, e) => s + e.X.rows, 0);
  const X = mat(n, d);
  const y = new Float64Array(n);
  let off = 0;
  for (const e of envs) {
    X.data.set(e.X.data, off * d);
    y.set(e.y, off);
    off += e.X.rows;
  }
  return { X, y };
}

function take(env: Env, idx: Int32Array): Env {
  const d = env.X.cols;
  const X = mat(idx.length, d);
  const y = new Float64Array(idx.length);
  idx.forEach((r, i) => {
    X.data.set(env.X.data.subarray(r * d, r * d + d), i * d);
    y[i] = env.y[r];
  });
  return { X, y };
}

export class ErmTrainer implements Trainer {
  readonly kind = "erm";
  steps = 0;
  W: Mat;
  V: Float64Array;
  private opt: Adam;
  private train: Env;

  constructor(trainEnvs: Env[], private cfg: BaseConfig, uniform: () => number) {
    this.train = pool(trainEnvs);
    ({ W: this.W, V: this.V } = initMlp(this.train.X.cols, cfg.hidden, uniform));
    this.opt = new Adam([this.W.data, this.V], cfg.lr);
  }

  step() {
    const { X, y } = this.train;
    const fw = forward(X, this.W, this.V);
    const g = nllGradients(X, y, fw, this.V);
    addScaled(g.dW.data, this.W.data, this.cfg.l2 * 2);
    addScaled(g.dV, this.V, this.cfg.l2 * 2);
    this.opt.step([g.dW.data, g.dV], ++this.steps);
  }

  logits(X: Mat) {
    return forward(X, this.W, this.V).F;
  }

  reliance(X: Mat) {
    return inputSensitivity(X, [this.W], [this.V]);
  }
}

export interface IrmConfig extends BaseConfig {
  penaltyWeight: number;
  annealIters: number;
}

export class IrmTrainer implements Trainer {
  readonly kind = "irm";
  steps = 0;
  W: Mat;
  V: Float64Array;
  private opt: Adam;

  constructor(private trainEnvs: Env[], private cfg: IrmConfig, uniform: () => number) {
    ({ W: this.W, V: this.V } = initMlp(trainEnvs[0].X.cols, cfg.hidden, uniform));
    this.opt = new Adam([this.W.data, this.V], cfg.lr);
  }

  /** Penalty weight in force at the next step (annealed in after `annealIters`). */
  get penaltyWeight() {
    return this.steps >= this.cfg.annealIters ? this.cfg.penaltyWeight : 1;
  }

  step() {
    const pw = this.penaltyWeight;
    const gW = new Float64Array(this.W.data.length);
    const gV = new Float64Array(this.V.length);
    for (const { X, y } of this.trainEnvs) {
      const fw = forward(X, this.W, this.V);
      const nll = nllGradients(X, y, fw, this.V);
      const pen = irmPenaltyGradients(X, y, fw, this.V);
      for (let i = 0; i < gW.length; i++) gW[i] += nll.dW.data[i] + pw * pen.dW.data[i];
      for (let i = 0; i < gV.length; i++) gV[i] += nll.dV[i] + pw * pen.dV[i];
    }
    const n = this.trainEnvs.length;
    for (let i = 0; i < gW.length; i++) gW[i] = gW[i] / n + this.cfg.l2 * 2 * this.W.data[i];
    for (let i = 0; i < gV.length; i++) gV[i] = gV[i] / n + this.cfg.l2 * 2 * this.V[i];
    if (pw > 1) {
      for (let i = 0; i < gW.length; i++) gW[i] /= pw;
      for (let i = 0; i < gV.length; i++) gV[i] /= pw;
    }
    this.opt.step([gW, gV], ++this.steps);
  }

  logits(X: Mat) {
    return forward(X, this.W, this.V).F;
  }

  reliance(X: Mat) {
    return inputSensitivity(X, [this.W], [this.V]);
  }
}

export interface GamesConfig extends BaseConfig {
  batchSize: number;
  warmStart: boolean;
}

export class IrmGamesTrainer implements Trainer {
  readonly kind = "irm_games";
  steps = 0;
  Ws: Mat[] = [];
  Vs: Float64Array[] = [];
  /** Player who moves on the next step. */
  turn = 0;
  private opts: Adam[];
  private tPlayer: number[];

  constructor(
    private trainEnvs: Env[],
    private cfg: GamesConfig,
    uniform: () => number,
    private sample: BatchSampler,
  ) {
    const d = trainEnvs[0].X.cols;
    // Python draws all players' W (stacked) before any V.
    const lw = Math.sqrt(6 / (d + cfg.hidden));
    const lv = Math.sqrt(6 / (cfg.hidden + 1));
    for (let e = 0; e < trainEnvs.length; e++) {
      const W = mat(cfg.hidden, d);
      for (let q = 0; q < W.data.length; q++) W.data[q] = -lw + 2 * lw * uniform();
      this.Ws.push(W);
    }
    for (let e = 0; e < trainEnvs.length; e++) {
      this.Vs.push(new Float64Array(cfg.hidden).map(() => -lv + 2 * lv * uniform()));
    }
    this.opts = this.Ws.map((W, e) => new Adam([W.data, this.Vs[e]], cfg.lr));
    this.tPlayer = this.Ws.map(() => 0);
    if (cfg.warmStart) this.warmStart();
  }

  private batch(e: number): Env {
    const env = this.trainEnvs[e];
    return take(env, this.sample(env.X.rows, Math.min(this.cfg.batchSize, env.X.rows)));
  }

  private warmStart() {
    const total = this.trainEnvs.reduce((s, e) => s + e.X.rows, 0);
    const rounds = Math.max(1, Math.floor(total / this.cfg.batchSize));
    for (let r = 0; r < rounds; r++) {
      for (let e = 0; e < this.Ws.length; e++) {
        const { X, y } = this.batch(e);
        const fw = forward(X, this.Ws[e], this.Vs[e]);
        const g = nllGradients(X, y, fw, this.Vs[e]);
        addScaled(g.dW.data, this.Ws[e].data, 2 * this.cfg.l2);
        addScaled(g.dV, this.Vs[e], 2 * this.cfg.l2);
        this.opts[e].step([g.dW.data, g.dV], ++this.tPlayer[e]);
      }
    }
  }

  step() {
    const e = this.turn;
    const { X, y } = this.batch(e);
    const fwSelf = forward(X, this.Ws[e], this.Vs[e]);
    const ens = ensembleLogits(X, this.Ws, this.Vs);
    const g = ensembleMemberNllGradients(X, y, fwSelf, ens, this.Vs[e], this.Ws.length);
    addScaled(g.dW.data, this.Ws[e].data, 2 * this.cfg.l2);
    addScaled(g.dV, this.Vs[e], 2 * this.cfg.l2);
    this.opts[e].step([g.dW.data, g.dV], ++this.tPlayer[e]);
    this.steps++;
    this.turn = (e + 1) % this.Ws.length;
  }

  logits(X: Mat) {
    return ensembleLogits(X, this.Ws, this.Vs);
  }

  reliance(X: Mat) {
    return inputSensitivity(X, this.Ws, this.Vs);
  }
}
