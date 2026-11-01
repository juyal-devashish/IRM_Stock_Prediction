import { useState } from "react";
import { pct } from "../format";
import { useWidth } from "../hooks";
import { METHOD_INFO, METHOD_ORDER } from "../methods";
import type { LiveSnapshot, LiveTicker, Variant } from "../types";

const VARIANTS: { key: Variant; label: string; note: string }[] = [
  {
    key: "shortcut_flipped",
    label: "Shortcut flipped",
    note: "The shortcut feature agrees with the label only 10% of the time, as in the report's test set.",
  },
  {
    key: "no_shortcut",
    label: "No shortcut",
    note: "The shortcut is a coin flip, so only the real price signal can help.",
  },
];

const TICKER_LABELS: Record<string, string> = { SPY: "S&P 500 (SPY)" };

const TREND = { bull: "bull market", bear: "bear market", side: "sideways market", unknown: "unclear trend" };

export function Live({ history }: { history: LiveSnapshot[] }) {
  const latest = history[history.length - 1];
  const tickers = Object.keys(latest.tickers);
  const [ticker, setTicker] = useState(tickers[0]);
  const [variant, setVariant] = useState<Variant>("shortcut_flipped");
  const t = latest.tickers[ticker] ?? latest.tickers[tickers[0]];
  const v = VARIANTS.find((x) => x.key === variant)!;

  return (
    <section id="live" aria-labelledby="h-live">
      <h2 id="h-live">Since 2024, on data the models never saw</h2>
      <p>
        The models were trained once on {latest.train.start.slice(0, 4)}–{latest.train.end.slice(0, 4)} prices and
        frozen. Every Monday a scheduled job scores them on every trading day since {latest.train.end}. Data runs to{" "}
        {t.as_of}, {t.n_days} trading days.
      </p>

      <div className="controls" role="group" aria-label="Choose a ticker">
        {tickers.map((k) => (
          <button key={k} type="button" className="chip small" aria-pressed={k === ticker} onClick={() => setTicker(k)}>
            {TICKER_LABELS[k] ?? k}
          </button>
        ))}
      </div>
      <div className="controls sub" role="group" aria-label="Choose a test variant">
        {VARIANTS.map((x) => (
          <button
            key={x.key}
            type="button"
            className="chip small"
            aria-pressed={x.key === variant}
            onClick={() => setVariant(x.key)}
          >
            {x.label}
          </button>
        ))}
      </div>

      <p className="regime">
        {TICKER_LABELS[ticker] ?? ticker} now: {TREND[t.regime.trend]}, {t.regime.volatility} volatility compared with 2010–2023.
      </p>

      <div className="panel">
        <MonthlyChart data={t} variant={variant} />
      </div>
      <p className="reading">
        {v.note} IRM and IRM-Games mostly sit on the "always up" line: they predict "up" almost every day, so their
        lines often overlap. They ignore the shortcut, but they have no edge over always predicting up. This is not a
        trading signal.
      </p>

      <div className="table-wrap">
        <table>
          <caption>
            {v.label}: accuracy on all {t.n_days} trading days since {latest.train.end}. "Always up" is the share of
            days the price rose.
          </caption>
          <thead>
            <tr>
              <th scope="col">Ticker</th>
              <th scope="col" className="num">Days</th>
              <th scope="col" className="num">Always up</th>
              {METHOD_ORDER.map((m) => (
                <th key={m} scope="col" className="num">
                  {METHOD_INFO[m].label}
                </th>
              ))}
              <th scope="col">Regime now</th>
            </tr>
          </thead>
          <tbody>
            {tickers.map((k) => {
              const row = latest.tickers[k];
              return (
                <tr key={k}>
                  <th scope="row">{k}</th>
                  <td className="num">{row.n_days}</td>
                  <td className="num">{pct(row.base_rate)}</td>
                  {METHOD_ORDER.map((m) => (
                    <td key={m} className="num">
                      {pct(row.variants[variant][m])}
                    </td>
                  ))}
                  <td>
                    {TREND[row.regime.trend]}, {row.regime.volatility} vol.
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="fine">
        Updated {latest.run_date} ({history.length} weekly {history.length === 1 ? "snapshot" : "snapshots"}).
        {Object.keys(latest.errors).length > 0 && ` Not updated this week: ${Object.keys(latest.errors).join(", ")}.`}
      </p>
    </section>
  );
}

const TOP = 16;
const BOTTOM = 30;
const LEFT = 40;
const RIGHT = 12;
const H = 280;

function MonthlyChart({ data, variant }: { data: LiveTicker; variant: Variant }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [idx, setIdx] = useState<number | null>(null);
  const rows = data.monthly;
  const w = Math.max(width, 300);
  const innerW = w - LEFT - RIGHT;
  const innerH = H - TOP - BOTTOM;
  const x = (i: number) => LEFT + (rows.length === 1 ? innerW / 2 : (i / (rows.length - 1)) * innerW);
  const y = (v: number) => TOP + (1 - v) * innerH;
  const path = (vals: number[]) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");

  const pick = (clientX: number, el: SVGSVGElement) => {
    const left = el.getBoundingClientRect().left;
    const i = Math.round(((clientX - left - LEFT) / innerW) * (rows.length - 1));
    setIdx(Math.max(0, Math.min(rows.length - 1, i)));
  };
  const active = idx === null ? null : rows[idx];

  return (
    <div ref={ref} className="line-chart">
      <ul className="legend" aria-hidden="true">
        {METHOD_ORDER.map((m) => (
          <li key={m}>
            <span className="swatch" style={{ background: `var(--m-${m})` }} />
            {METHOD_INFO[m].label}
          </li>
        ))}
        <li>
          <span className="swatch baseline-swatch" />
          Always up
        </li>
      </ul>
      <svg
        width={w}
        height={H}
        tabIndex={0}
        role="img"
        aria-label={`Monthly accuracy since 2024. Use left and right arrow keys to read each month.`}
        onPointerMove={(e) => pick(e.clientX, e.currentTarget)}
        onPointerLeave={() => setIdx(null)}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") setIdx((i) => Math.min(rows.length - 1, (i ?? -1) + 1));
          if (e.key === "ArrowLeft") setIdx((i) => Math.max(0, (i ?? rows.length) - 1));
        }}
        onBlur={() => setIdx(null)}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className={t === 0.5 ? "chance" : "grid"} x1={LEFT} x2={w - RIGHT} y1={y(t)} y2={y(t)} />
            <text className="tick" x={LEFT - 8} y={y(t) + 4} textAnchor="end">
              {t * 100}%
            </text>
          </g>
        ))}
        {rows.map((r, i) =>
          r.month.endsWith("-01") ? (
            <text key={r.month} className="tick" x={x(i)} y={H - 8} textAnchor="middle">
              {r.month.slice(0, 4)}
            </text>
          ) : null,
        )}
        <path className="baseline-line" d={path(rows.map((r) => r.base_rate))} />
        {METHOD_ORDER.map((m) => (
          <path key={m} className="series" style={{ stroke: `var(--m-${m})` }} d={path(rows.map((r) => r[variant][m]))} />
        ))}
        {active && idx !== null && (
          <g className="crosshair">
            <line x1={x(idx)} x2={x(idx)} y1={TOP} y2={H - BOTTOM} />
            {METHOD_ORDER.map((m) => (
              <circle key={m} cx={x(idx)} cy={y(active[variant][m])} r={4.5} style={{ fill: `var(--m-${m})` }} />
            ))}
          </g>
        )}
      </svg>
      <p className="readout" aria-live="polite">
        {active
          ? `${active.month} (${active.n_days} days): always up ${pct(active.base_rate)}, ` +
            METHOD_ORDER.map((m) => `${METHOD_INFO[m].label} ${pct(active[variant][m])}`).join(", ")
          : "Hover, or focus the chart and use the arrow keys, to read a month."}
      </p>
    </div>
  );
}
