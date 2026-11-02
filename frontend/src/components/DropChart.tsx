import { useState } from "react";
import { pct, points } from "../format";
import { useWidth } from "../hooks";
import { CHART_METHODS, METHOD_INFO } from "../methods";
import type { MethodKey, Run } from "../types";

const ROW_H = 76;
const TOP = 34;
const BOTTOM = 34;
const PAD_X = 20;
const TICKS = [0, 0.25, 0.5, 0.75, 1];

/** Train → test accuracy per method on one 0-100% axis, with the coin-flip line at 50%. */
export function DropChart({ run }: { run: Run }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [active, setActive] = useState<MethodKey | null>(null);
  const w = Math.max(width, 300);
  const innerW = w - PAD_X * 2;
  const x = (v: number) => PAD_X + v * innerW;
  const h = TOP + ROW_H * CHART_METHODS.length + BOTTOM;
  const activeStats = active ? run.methods[active] : null;

  return (
    <div ref={ref} className="drop-chart">
      <svg width={w} height={h} role="group" aria-label="Train and test accuracy by method">
        {TICKS.map((t) => (
          <g key={t}>
            <line className="grid" x1={x(t)} x2={x(t)} y1={TOP - 8} y2={h - BOTTOM + 4} />
            <text className="tick" x={x(t)} y={h - 10} textAnchor="middle">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        <line className="chance" x1={x(0.5)} x2={x(0.5)} y1={TOP - 14} y2={h - BOTTOM + 4} />
        <text className="chance-label" x={x(0.5) + 6} y={TOP - 16}>
          Coin flip
        </text>

        {CHART_METHODS.map((key, i) => {
          const s = run.methods[key];
          const y = TOP + ROW_H * i + ROW_H / 2 + 8;
          const close = Math.abs(x(s.train_mean) - x(s.test_mean)) < 84;
          const trainRight = s.train_mean >= s.test_mean;
          return (
            <g
              key={key}
              className={`row ${active === key ? "is-active" : ""}`}
              style={{ ["--m" as string]: `var(--m-${key})` }}
              tabIndex={0}
              role="img"
              aria-label={`${METHOD_INFO[key].label}: train ${pct(s.train_mean)}, test ${pct(s.test_mean)}`}
              onMouseEnter={() => setActive(key)}
              onMouseLeave={() => setActive(null)}
              onFocus={() => setActive(key)}
              onBlur={() => setActive(null)}
            >
              <rect className="hit" x={0} y={y - ROW_H / 2 + 4} width={w} height={ROW_H - 8} />
              <text className="method-label" x={PAD_X} y={y - 26}>
                {METHOD_INFO[key].label}
              </text>
              <line
                className="link"
                x1={x(s.train_mean)}
                x2={x(s.test_mean)}
                y1={y}
                y2={y}
              />
              <circle className="dot train" cx={x(s.train_mean)} cy={y} r={7} />
              <circle className="dot test" cx={x(s.test_mean)} cy={y} r={7} />
              <text
                className="value"
                x={close ? x(s.train_mean) : x(s.train_mean) + (trainRight ? 14 : -14)}
                y={close ? y - 14 : y + 4}
                textAnchor={close ? "middle" : trainRight ? "start" : "end"}
              >
                {pct(s.train_mean)}
              </text>
              <text
                className="value strong"
                x={close ? x(s.test_mean) : x(s.test_mean) + (trainRight ? -14 : 14)}
                y={close ? y + 26 : y + 4}
                textAnchor={close ? "middle" : trainRight ? "end" : "start"}
              >
                {pct(s.test_mean)}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="chart-foot">
        <span className="key" aria-hidden="true">
          <svg width="14" height="14"><circle className="key-train" cx="7" cy="7" r="5.5" /></svg>
          Train
        </span>
        <span className="key" aria-hidden="true">
          <svg width="14" height="14"><circle className="key-test" cx="7" cy="7" r="5.5" /></svg>
          Test (shortcut flipped)
        </span>
        <span className="readout" aria-live="polite">
          {active && activeStats
            ? `${METHOD_INFO[active].label}: ${pct(activeStats.train_mean)} ± ${pct(activeStats.train_std)} → ${pct(activeStats.test_mean)} ± ${pct(activeStats.test_std)} (drop of ${points(activeStats.train_mean - activeStats.test_mean)})`
            : "Hover or focus a row for spread and drop."}
        </span>
      </div>
    </div>
  );
}
