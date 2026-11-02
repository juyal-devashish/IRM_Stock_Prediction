import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { pct } from "../format";
import { useWidth } from "../hooks";
import { METHOD_INFO } from "../methods";
import type { MethodKey } from "../types";
import { type Env, ErmTrainer, IrmTrainer, type Trainer } from "./methods";
import { Tex } from "../components/Tex";
import { CHART_METHODS } from "../methods";
import { mat, meanAccuracy } from "./nn";
import { makeToyEnv, mulberry32, withShortcutAgreement } from "./toy";

// Settings chosen so the effect is visible in a few seconds (see PR #4 notes).
const BASE = { hidden: 16, lr: 0.02, l2: 1e-3 };
const IRM = { penaltyWeight: 1e4, annealIters: 100 };
const N_TRAIN = 500;
const N_TEST = 1000;
const MAX_ROUNDS = 1500;
const ROUNDS_PER_FRAME = 5;
const CANONICAL_AGREE = 0.1;
const SWEEP = Array.from({ length: 11 }, (_, i) => i / 10);
const X_RANGE = [-5.5, 5.5] as const;
const S_RANGE = [-1.7, 1.7] as const;

interface World {
  train: Env[];
  testBase: Env;
  flips: Float64Array;
  canonical: Env;
  trainers: Trainer[];
}

function buildWorld(seed: number, agree: [number, number]): World {
  const rand = mulberry32(seed);
  const train = [makeToyEnv(N_TRAIN, agree[0], rand), makeToyEnv(N_TRAIN, agree[1], rand)];
  const testBase = makeToyEnv(N_TEST, 1, rand);
  const flips = Float64Array.from({ length: N_TEST }, rand);
  return {
    train,
    testBase,
    flips,
    canonical: withShortcutAgreement(testBase, flips, CANONICAL_AGREE),
    trainers: [new ErmTrainer(train, BASE, rand), new IrmTrainer(train, { ...BASE, ...IRM }, rand)],
  };
}

/** Share of the model's sensitivity that comes from the shortcut, scaled by each feature's spread. */
function shortcutShare(t: Trainer, env: Env): number {
  const g = t.reliance(env.X);
  const sd = [0, 1].map((k) => {
    let m = 0;
    let v = 0;
    for (let i = 0; i < env.X.rows; i++) m += env.X.data[2 * i + k];
    m /= env.X.rows;
    for (let i = 0; i < env.X.rows; i++) v += (env.X.data[2 * i + k] - m) ** 2;
    return Math.sqrt(v / env.X.rows);
  });
  const a = g[0] * sd[0];
  const b = g[1] * sd[1];
  return a + b === 0 ? 0 : b / (a + b);
}

const trainAcc = (t: Trainer, envs: Env[]) =>
  envs.reduce((s, e) => s + meanAccuracy(t.logits(e.X), e.y), 0) / envs.length;

interface HistoryPoint {
  round: number;
  acc: Record<MethodKey, number>;
}

export function Lab({ theme }: { theme: string }) {
  const [seed, setSeed] = useState(42);
  const [agree, setAgree] = useState<[number, number]>([0.9, 0.8]);
  const [dial, setDial] = useState(CANONICAL_AGREE);
  const [playing, setPlaying] = useState(false);
  const [, setTick] = useState(0);
  const world = useRef<World>(buildWorld(seed, agree));
  const round = useRef(0);
  const history = useRef<HistoryPoint[]>([]);

  const record = useCallback(() => {
    const w = world.current;
    const acc = Object.fromEntries(
      w.trainers.map((t) => [t.kind, meanAccuracy(t.logits(w.canonical.X), w.canonical.y)]),
    ) as Record<MethodKey, number>;
    history.current.push({ round: round.current, acc });
  }, []);

  const reset = useCallback(
    (nextSeed = seed, nextAgree = agree) => {
      setPlaying(false);
      world.current = buildWorld(nextSeed, nextAgree);
      round.current = 0;
      history.current = [];
      record();
      setTick((n) => n + 1);
    },
    [seed, agree, record],
  );

  // Initial point on the curve.
  useEffect(() => {
    if (history.current.length === 0) record();
    setTick((n) => n + 1);
  }, [record]);

  const advance = useCallback(
    (rounds: number) => {
      const w = world.current;
      for (let r = 0; r < rounds && round.current < MAX_ROUNDS; r++) {
        for (const t of w.trainers) t.step();
        round.current++;
      }
      record();
      if (round.current >= MAX_ROUNDS) setPlaying(false);
      setTick((n) => n + 1);
    },
    [record],
  );

  useEffect(() => {
    if (!playing) return;
    let id = 0;
    const loop = () => {
      advance(ROUNDS_PER_FRAME);
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [playing, advance]);

  const w = world.current;
  const testEnv = useMemo(
    () => withShortcutAgreement(w.testBase, w.flips, dial),
    // Recompute when the world is rebuilt or the dial moves.
    [w, dial],
  );
  const done = round.current >= MAX_ROUNDS;

  return (
    <section id="lab" aria-labelledby="h-lab">
      <h2 id="h-lab">Train it yourself</h2>
      <p>
        A toy version of the problem, trained live in your browser with the same algorithms. The signal feature is
        right about 75% of the time in every environment. The shortcut is right {pct(agree[0], 0)} and{" "}
        {pct(agree[1], 0)} of the time in the two training environments, so it looks better than the signal.
      </p>

      <div className="math-block lab-math">
        <Tex display>
          {"x_{\\text{signal}} = (2y - 1) + \\varepsilon, \\quad \\varepsilon \\sim \\mathcal{N}(0, 1.5^2) \\qquad x_{\\text{shortcut}} = (2\\tilde y - 1) + \\eta, \\quad \\eta \\sim \\mathcal{N}(0, 0.1^2)"}
        </Tex>
        <p className="fine">
          <Tex>{"\\tilde y = y"}</Tex> with probability <Tex>{"a_e"}</Tex> and <Tex>{"1 - y"}</Tex> otherwise. On its
          own the signal is right <Tex>{"\\Phi(1/1.5) \\approx 75\\%"}</Tex> of the time. "Uses the shortcut" is
          the shortcut's share of the model's average input sensitivity, with each feature scaled by its spread:
        </p>
        <Tex display>
          {"\\dfrac{\\overline{|\\partial f / \\partial x_{\\text{sc}}|}\\,\\sigma_{\\text{sc}}}{\\overline{|\\partial f / \\partial x_{\\text{sig}}|}\\,\\sigma_{\\text{sig}} + \\overline{|\\partial f / \\partial x_{\\text{sc}}|}\\,\\sigma_{\\text{sc}}}"}
        </Tex>
      </div>

      <div className="lab-controls">
        <button
          type="button"
          className="chip primary"
          onClick={() => (done ? reset() : setPlaying((p) => !p))}
          aria-pressed={playing}
        >
          {playing ? "Pause" : done ? "Train again" : round.current === 0 ? "Start training" : "Resume"}
        </button>
        <button type="button" className="chip" onClick={() => advance(25)} disabled={playing || done}>
          Step 25 rounds
        </button>
        <button
          type="button"
          className="chip"
          onClick={() => {
            setSeed(seed + 1);
            reset(seed + 1);
          }}
        >
          New data
        </button>
        <span className="lab-round" aria-live="polite">
          Round {round.current} of {MAX_ROUNDS}
          {round.current >= IRM.annealIters ? ", IRM penalty on" : ""}
        </span>
      </div>

      <div className="lab-sliders">
        {([0, 1] as const).map((i) => (
          <label key={i}>
            <span>
              Training environment {i + 1}: shortcut right <b>{pct(agree[i], 0)}</b>
            </span>
            <input
              type="range"
              min={0.5}
              max={1}
              step={0.05}
              value={agree[i]}
              onChange={(e) => {
                const next: [number, number] = [...agree] as [number, number];
                next[i] = Number(e.target.value);
                setAgree(next);
                reset(seed, next);
              }}
            />
          </label>
        ))}
      </div>

      <div className="lab-panels">
        {w.trainers.map((t) => (
          <BoundaryPanel
            key={t.kind}
            trainer={t}
            testEnv={testEnv}
            train={w.train}
            theme={theme}
            tick={round.current}
          />
        ))}
      </div>

      <div className="lab-dial">
        <label htmlFor="dial">
          <span>
            Test environment: the shortcut is right <b>{pct(dial, 0)}</b> of the time
          </span>
        </label>
        <input
          id="dial"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={dial}
          onChange={(e) => setDial(Number(e.target.value))}
        />
        <p className="fine">
          Moving this only changes the test data, so the models do not retrain. A model that leans on the shortcut
          rises and falls with it. One that ignores it stays flat.
        </p>
      </div>

      <div className="lab-charts">
        <div className="panel">
          <h3 className="chart-title">Test accuracy while training (shortcut right {pct(CANONICAL_AGREE, 0)})</h3>
          <CurveChart history={history.current} />
        </div>
        <div className="panel">
          <h3 className="chart-title">Test accuracy as the shortcut changes (current models)</h3>
          <SweepChart trainers={w.trainers} world={w} dial={dial} tick={round.current} />
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- boundary panel

const GRID_W = 96;
const GRID_H = 60;

function BoundaryPanel({
  trainer,
  testEnv,
  train,
  theme,
  tick,
}: {
  trainer: Trainer;
  testEnv: Env;
  train: Env[];
  theme: string;
  tick: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const grid = useMemo(() => {
    const g = mat(GRID_W * GRID_H, 2);
    for (let r = 0; r < GRID_H; r++) {
      for (let c = 0; c < GRID_W; c++) {
        const i = r * GRID_W + c;
        g.data[2 * i] = X_RANGE[0] + ((c + 0.5) / GRID_W) * (X_RANGE[1] - X_RANGE[0]);
        g.data[2 * i + 1] = S_RANGE[1] - ((r + 0.5) / GRID_H) * (S_RANGE[1] - S_RANGE[0]);
      }
    }
    return g;
  }, []);

  const testAcc = meanAccuracy(trainer.logits(testEnv.X), testEnv.y);
  const tr = trainAcc(trainer, train);
  const share = shortcutShare(trainer, testEnv);
  const cssW = Math.max(width, 200);
  const cssH = Math.round(cssW * 0.62);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = cssW * dpr;
    el.height = cssH * dpr;
    const ctx = el.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink").trim();
    const surface = css.getPropertyValue("--surface").trim();
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, cssW, cssH);

    const logits = trainer.logits(grid);
    const cw = cssW / GRID_W;
    const ch = cssH / GRID_H;
    // Shade the "label 1" region. Integer cell edges avoid darker seams where cells overlap.
    ctx.globalAlpha = 0.14;
    ctx.fillStyle = ink;
    for (let r = 0; r < GRID_H; r++) {
      const y0 = Math.floor(r * ch);
      const y1 = Math.floor((r + 1) * ch);
      for (let c = 0; c < GRID_W; c++) {
        if (logits[r * GRID_W + c] <= 0) continue;
        const x0 = Math.floor(c * cw);
        ctx.fillRect(x0, y0, Math.floor((c + 1) * cw) - x0, y1 - y0);
      }
    }
    // Decision boundary: cells whose sign differs from the right or lower neighbour.
    ctx.globalAlpha = 0.9;
    for (let r = 0; r < GRID_H; r++) {
      for (let c = 0; c < GRID_W; c++) {
        const s = logits[r * GRID_W + c] > 0;
        const right = c + 1 < GRID_W && logits[r * GRID_W + c + 1] > 0 !== s;
        const down = r + 1 < GRID_H && logits[(r + 1) * GRID_W + c] > 0 !== s;
        if (right) ctx.fillRect((c + 1) * cw - 1, r * ch, 2, ch);
        if (down) ctx.fillRect(c * cw, (r + 1) * ch - 1, cw, 2);
      }
    }
    // A sample of test points: filled = label 1, ring = label 0.
    const sx = (x: number) => ((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])) * cssW;
    const sy = (s: number) => ((S_RANGE[1] - s) / (S_RANGE[1] - S_RANGE[0])) * cssH;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = 1.25;
    ctx.strokeStyle = ink;
    for (let i = 0; i < 160; i++) {
      const x = sx(testEnv.X.data[2 * i]);
      const s = sy(testEnv.X.data[2 * i + 1] + ((i * 37) % 11 - 5) * 0.025); // tiny jitter
      ctx.beginPath();
      ctx.arc(x, s, 2.6, 0, Math.PI * 2);
      if (testEnv.y[i] === 1) ctx.fill();
      else ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }, [trainer, grid, testEnv, cssW, cssH, theme, tick]);

  const info = METHOD_INFO[trainer.kind];
  return (
    <figure className="boundary" style={{ ["--m" as string]: `var(--m-${trainer.kind})` }}>
      <figcaption>
        <span className="swatch" style={{ background: `var(--m-${trainer.kind})` }} />
        <b>{info.label}</b>
      </figcaption>
      <div ref={ref} className="canvas-wrap">
        <canvas
          ref={canvas}
          style={{ width: "100%", height: cssH }}
          role="img"
          aria-label={`${info.label} decision boundary. Test accuracy ${pct(testAcc)}; ${pct(share, 0)} of its sensitivity comes from the shortcut.`}
        />
        <span className="axis-x">Signal →</span>
        <span className="axis-y">Shortcut →</span>
      </div>
      <dl className="boundary-stats">
        <div>
          <dt>Train</dt>
          <dd>{pct(tr)}</dd>
        </div>
        <div>
          <dt>Test</dt>
          <dd className={testAcc < 0.5 ? "below" : undefined}>{pct(testAcc)}</dd>
        </div>
        <div className="share">
          <dt>Uses the shortcut</dt>
          <dd>
            <span className="share-bar" aria-hidden="true">
              <span style={{ width: `${share * 100}%` }} />
            </span>
            {pct(share, 0)}
          </dd>
        </div>
      </dl>
    </figure>
  );
}

// ---------------------------------------------------------------------- charts

const ORDER: MethodKey[] = CHART_METHODS;
const CH = 200;
const PAD = { l: 40, r: 12, t: 12, b: 26 };

function Axes({ w, xLabels }: { w: number; xLabels: [number, string][] }) {
  const ih = CH - PAD.t - PAD.b;
  return (
    <>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line
            className={t === 0.5 ? "chance" : "grid"}
            x1={PAD.l}
            x2={w - PAD.r}
            y1={PAD.t + (1 - t) * ih}
            y2={PAD.t + (1 - t) * ih}
          />
          <text className="tick" x={PAD.l - 8} y={PAD.t + (1 - t) * ih + 4} textAnchor="end">
            {t * 100}%
          </text>
        </g>
      ))}
      {xLabels.map(([x, label], i) => (
        <text
          key={label}
          className="tick"
          x={x}
          y={CH - 6}
          textAnchor={i === 0 ? "start" : i === xLabels.length - 1 ? "end" : "middle"}
        >
          {label}
        </text>
      ))}
    </>
  );
}

function Legend() {
  return (
    <ul className="legend" aria-hidden="true">
      {ORDER.map((m) => (
        <li key={m}>
          <span className="swatch" style={{ background: `var(--m-${m})` }} />
          {METHOD_INFO[m].label}
        </li>
      ))}
    </ul>
  );
}

function CurveChart({ history }: { history: HistoryPoint[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const w = Math.max(width, 260);
  const iw = w - PAD.l - PAD.r;
  const ih = CH - PAD.t - PAD.b;
  const x = (r: number) => PAD.l + (r / MAX_ROUNDS) * iw;
  const y = (v: number) => PAD.t + (1 - v) * ih;
  const last = history[history.length - 1];
  return (
    <div ref={ref} className="line-chart">
      <Legend />
      <svg width={w} height={CH} role="img" aria-label={
        last ? `After ${last.round} rounds: ` + ORDER.map((m) => `${METHOD_INFO[m].label} ${pct(last.acc[m])}`).join(", ") : "No training yet"
      }>
        <Axes w={w} xLabels={[[x(0), "0"], [x(MAX_ROUNDS / 2), `${MAX_ROUNDS / 2}`], [x(MAX_ROUNDS), `${MAX_ROUNDS} rounds`]]} />
        <line className="annotation" x1={x(IRM.annealIters)} x2={x(IRM.annealIters)} y1={PAD.t} y2={CH - PAD.b} />
        <text className="annotation-label" x={x(IRM.annealIters) + 4} y={PAD.t + 10}>
          IRM penalty on
        </text>
        {ORDER.map((m) => (
          <path
            key={m}
            className="series"
            style={{ stroke: `var(--m-${m})` }}
            d={history.map((h, i) => `${i ? "L" : "M"}${x(h.round).toFixed(1)},${y(h.acc[m]).toFixed(1)}`).join("")}
          />
        ))}
      </svg>
    </div>
  );
}

function SweepChart({ trainers, world, dial, tick }: { trainers: Trainer[]; world: World; dial: number; tick: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const w = Math.max(width, 260);
  const iw = w - PAD.l - PAD.r;
  const ih = CH - PAD.t - PAD.b;
  const x = (a: number) => PAD.l + a * iw;
  const y = (v: number) => PAD.t + (1 - v) * ih;
  // Recompute at most every few rounds while training; always when the dial or world changes.
  const bucket = Math.floor(tick / 10);
  const curves = useMemo(() => {
    const envs = SWEEP.map((a) => withShortcutAgreement(world.testBase, world.flips, a));
    return trainers.map((t) => SWEEP.map((_, i) => meanAccuracy(t.logits(envs[i].X), envs[i].y)));
  }, [trainers, world, bucket]);
  return (
    <div ref={ref} className="line-chart">
      <Legend />
      <svg width={w} height={CH} role="img" aria-label="Test accuracy of each current model as the shortcut's agreement with the label goes from 0% to 100%">
        <Axes w={w} xLabels={[[x(0), "0%"], [x(0.5), "50%"], [x(1), "100% agreement"]]} />
        <line className="dial-marker" x1={x(dial)} x2={x(dial)} y1={PAD.t} y2={CH - PAD.b} />
        {curves.map((c, k) => (
          <path
            key={trainers[k].kind}
            className="series"
            style={{ stroke: `var(--m-${trainers[k].kind})` }}
            d={c.map((v, i) => `${i ? "L" : "M"}${x(SWEEP[i]).toFixed(1)},${y(v).toFixed(1)}`).join("")}
          />
        ))}
      </svg>
    </div>
  );
}
