import { describe, expect, it } from "vitest";
import fixture from "./parity.fixture.json";
import { ErmTrainer, type Env, IrmGamesTrainer, IrmTrainer } from "./methods";
import {
  Adam,
  ensembleMemberNllGradients,
  forward,
  irmPenalty,
  irmPenaltyGradients,
  mat,
  meanAccuracy,
  meanNll,
  nllGradients,
} from "./nn";

const H = fixture.hidden;
const envs: Env[] = fixture.envs.map((e) => ({
  X: mat(e.rows, e.cols, e.X),
  y: Float64Array.from(e.y),
}));
const u = fixture.unit;

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, rtol: number, atol = 1e-12) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const tol = atol + rtol * Math.abs(expected[i]);
    if (Math.abs(actual[i] - expected[i]) > tol) {
      throw new Error(`index ${i}: ${actual[i]} vs ${expected[i]} (tol ${tol})`);
    }
  }
}

/** Replays a recorded stream of numbers (uniform draws or batch indices). */
function replay(values: number[]) {
  let i = 0;
  return () => {
    if (i >= values.length) throw new Error("replay stream exhausted");
    return values[i++];
  };
}

describe("nn core matches the Python core", () => {
  const X = envs[0].X;
  const y = envs[0].y;
  const W = mat(H, 2, u.W);
  const V = Float64Array.from(u.V);
  const fw = forward(X, W, V);

  it("forward, losses and accuracy", () => {
    close(fw.F, u.F, 1e-12);
    expect(meanNll(fw.F, y)).toBeCloseTo(u.nll, 12);
    expect(irmPenalty(fw.F, y)).toBeCloseTo(u.penalty, 12);
    expect(meanAccuracy(fw.F, y)).toBe(u.acc);
  });

  it("NLL, penalty and ensemble-member gradients", () => {
    const g = nllGradients(X, y, fw, V);
    close(g.dW.data, u.nll_dW, 1e-10);
    close(g.dV, u.nll_dV, 1e-10);
    const p = irmPenaltyGradients(X, y, fw, V);
    close(p.dW.data, u.pen_dW, 1e-10);
    close(p.dV, u.pen_dV, 1e-10);
    const F2 = forward(X, mat(H, 2, u.W2), Float64Array.from(u.V2)).F;
    const ens = fw.F.map((f, i) => (f + F2[i]) / 2);
    const e = ensembleMemberNllGradients(X, y, fw, ens, V, 2);
    close(e.dW.data, u.ens_dW, 1e-10);
    close(e.dV, u.ens_dV, 1e-10);
  });

  it("Adam", () => {
    const p = Float64Array.from(u.adam.p0);
    const opt = new Adam([p], u.adam.lr);
    opt.step([Float64Array.from(u.adam.g1)], 1);
    opt.step([Float64Array.from(u.adam.g2)], 2);
    close(p, u.adam.p2, 1e-12);
  });
});

describe("trainers match the Python training functions", () => {
  const [train0, train1, test] = envs;

  it("ERM", () => {
    const r = fixture.runs.erm;
    const t = new ErmTrainer([train0, train1], { hidden: H, lr: r.config.lr, l2: r.config.l2 }, replay(r.uniforms));
    for (let s = 0; s < r.config.steps; s++) t.step();
    close(t.W.data, r.W, 1e-9);
    close(t.V, r.V, 1e-9);
    close(t.logits(test.X), r.test_logits, 1e-9);
  });

  it("IRM with penalty annealing", () => {
    const r = fixture.runs.irm;
    const t = new IrmTrainer(
      [train0, train1],
      {
        hidden: H,
        lr: r.config.lr,
        l2: r.config.l2,
        penaltyWeight: r.config.penalty_weight,
        annealIters: r.config.penalty_anneal_iters,
      },
      replay(r.uniforms),
    );
    for (let s = 0; s < r.config.steps; s++) t.step();
    close(t.W.data, r.W, 1e-9);
    close(t.V, r.V, 1e-9);
    close(t.logits(test.X), r.test_logits, 1e-9);
  });

  it("IRM-Games with warm start (Python accumulates ensemble logits in float32)", () => {
    const r = fixture.runs.irm_games;
    const ints = replay(r.ints);
    const sampler = (_n: number, size: number) => Int32Array.from({ length: size }, () => ints());
    const t = new IrmGamesTrainer(
      [train0, train1],
      { hidden: H, lr: r.config.lr, l2: r.config.l2, batchSize: r.config.batch_size, warmStart: r.config.warm_start },
      replay(r.uniforms),
      sampler,
    );
    // One Python step is one round: every player takes a turn.
    for (let s = 0; s < r.config.steps * 2; s++) t.step();
    close(t.Ws.flatMap((w) => Array.from(w.data)), r.W, 1e-5, 1e-6);
    close(t.Vs.flatMap((v) => Array.from(v)), r.V, 1e-5, 1e-6);
    close(t.logits(test.X), r.test_logits, 1e-5, 1e-6);
  });
});
