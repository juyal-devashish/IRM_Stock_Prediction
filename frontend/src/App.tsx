import { useMemo, useState } from "react";
import { DropChart } from "./components/DropChart";
import { ResultsTable } from "./components/ResultsTable";
import { Shortcut } from "./components/Shortcut";
import { pct } from "./format";
import { useResults, useTheme } from "./hooks";
import { METHOD_INFO, METHOD_ORDER } from "./methods";
import type { Results } from "./types";

const REPO = "https://github.com/juyal-devashish/IRM_Stock_Prediction";

export default function App() {
  const { data, error } = useResults();
  const [theme, toggleTheme] = useTheme();

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#top">IRM Stock Prediction</a>
        <nav aria-label="Primary">
          <a href="#results">Results</a>
          <a href="#figures">Figures</a>
          <a href="#run">Run it</a>
          <a href={REPO}>Source</a>
          <button type="button" className="theme" onClick={toggleTheme} aria-pressed={theme === "dark"}>
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
        </nav>
      </header>
      <main id="top">
        {error && (
          <p role="alert" className="notice">
            The results data did not load: {error}. Run <code>uv run python scripts/build_dashboard_data.py</code> and
            reload.
          </p>
        )}
        {!data && !error && <p className="notice">Loading results…</p>}
        {data && <Dashboard data={data} />}
      </main>
      <footer>
        <p>
          Course project (ECE 553). Results come from the logs and saved arrays in{" "}
          <a href={`${REPO}/tree/main/results`}>results/</a>. Three finance logs without a recorded ticker, and one
          truncated run, are left out.
        </p>
      </footer>
    </>
  );
}

function Dashboard({ data }: { data: Results }) {
  const [benchId, setBenchId] = useState("time-split");
  const bench = data.benchmarks.find((b) => b.id === benchId) ?? data.benchmarks[0];
  const [runId, setRunId] = useState<string | null>(null);
  const run = bench.runs.find((r) => r.id === runId) ?? bench.runs[0];

  const headline = useMemo(() => {
    const erm = data.benchmarks.find((b) => b.id === "time-split")?.runs.find((r) => r.ticker === "AAPL")?.methods.erm;
    return erm
      ? `Standard training scores ${pct(erm.train_mean, 0)} on the stock history it learned from and ${pct(erm.test_mean, 0)} when the pattern shifts.`
      : "Standard training collapses when the pattern it learned shifts.";
  }, [data]);

  return (
    <>
      <section className="hero" aria-labelledby="h1">
        <h1 id="h1">{headline}</h1>
        <p className="lede">
          Below 50% is worse than a coin flip. The model leaned on a shortcut that held in training and reversed at
          test time. Invariant risk minimisation (IRM) is built to ignore shortcuts like that. This page compares ERM,
          IRM and IRM-Games on stock data, coloured digits and a synthetic benchmark.
        </p>

        <div className="controls" role="group" aria-label="Choose an experiment">
          {data.benchmarks.map((b) => (
            <button
              key={b.id}
              type="button"
              className="chip"
              aria-pressed={b.id === bench.id}
              onClick={() => {
                setBenchId(b.id);
                setRunId(null);
              }}
            >
              {b.label}
            </button>
          ))}
        </div>
        {bench.runs.length > 1 && (
          <div className="controls sub" role="group" aria-label="Choose a ticker">
            {bench.runs.map((r) => (
              <button
                key={r.id}
                type="button"
                className="chip small"
                aria-pressed={r.id === run.id}
                onClick={() => setRunId(r.id)}
              >
                {r.ticker}
              </button>
            ))}
          </div>
        )}

        <div className="panel">
          <DropChart run={run} />
        </div>
        <p className="reading">
          {bench.label === "ColoredMNIST"
            ? "On ColoredMNIST, ERM and IRM-Games fall to about 10%, far below chance, while IRM keeps 65%."
            : "On shifted stock data, IRM stays near 50%. That is chance level: it stops trusting the shortcut but does not find a stronger signal. IRM-Games varies by ticker."}
        </p>
      </section>

      <section id="results" aria-labelledby="h-why">
        <h2 id="h-why">What changes between training and test</h2>
        <div className="two-col">
          <div>
            <p>{bench.description}</p>
            <p>
              The shortcut is a feature that copies the label. Where it agrees with the label more often than not, a
              model can lean on it instead of the real signal. In the test environment it agrees only 10% of the time,
              so a model that relied on it gets most answers backwards.
            </p>
          </div>
          <div>
            <h3>Shortcut agrees with the label</h3>
            <Shortcut benchmarkId={bench.id} />
          </div>
        </div>

        <h3 className="spaced">The three methods</h3>
        <dl className="methods">
          {METHOD_ORDER.map((k) => (
            <div key={k}>
              <dt>
                <span className="swatch" style={{ background: `var(--m-${k})` }} />
                {METHOD_INFO[k].label}
              </dt>
              <dd>{METHOD_INFO[k].blurb}</dd>
            </div>
          ))}
        </dl>

        <h3 className="spaced">Numbers for {bench.label}</h3>
        <ResultsTable benchmark={bench} />
        {bench.id === "colored-mnist" && (
          <p className="fine">The saved IRM-Games result for ColoredMNIST has only 2 restarts, ERM and IRM have 10.</p>
        )}
      </section>

      <section id="figures" aria-labelledby="h-fig">
        <h2 id="h-fig">Figures from the report</h2>
        <div className="gallery">
          {data.figures.map((f) => (
            <figure key={f.file}>
              <a href={`${import.meta.env.BASE_URL}${f.file}`}>
                <img src={`${import.meta.env.BASE_URL}${f.file}`} alt={f.caption} loading="lazy" />
              </a>
              <figcaption>{f.caption}</figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section id="run" aria-labelledby="h-run">
        <h2 id="h-run">Run it yourself</h2>
        <p>
          Everything is NumPy, so there is no GPU requirement. Finance runs download prices from Yahoo Finance and need
          internet access.
        </p>
        <pre>
          <code>{`git clone ${REPO}.git
cd IRM_Stock_Prediction
uv sync --extra api
uv run irm-stock finance --experiment 2 --n-restarts 1 --steps 100 --games-steps 100
uv run irm-stock serve   # API docs at http://127.0.0.1:8000/docs`}</code>
        </pre>
        <p>
          The <a href={REPO}>repository</a> has the package, tests that check the code against the original research
          scripts, and the API.
        </p>
      </section>
    </>
  );
}
