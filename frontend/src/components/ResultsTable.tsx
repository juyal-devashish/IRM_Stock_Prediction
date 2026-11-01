import { pct, points } from "../format";
import { METHOD_INFO, METHOD_ORDER } from "../methods";
import type { Benchmark } from "../types";

export function ResultsTable({ benchmark }: { benchmark: Benchmark }) {
  return (
    <div className="table-wrap">
      <table>
        <caption>
          Accuracy, mean ± standard deviation across restarts. Gap is train minus test.
        </caption>
        <thead>
          <tr>
            <th scope="col">{benchmark.runs[0].ticker ? "Ticker" : "Run"}</th>
            <th scope="col">Method</th>
            <th scope="col" className="num">Train</th>
            <th scope="col" className="num">Test</th>
            <th scope="col" className="num">Gap</th>
            <th scope="col" className="num">Restarts</th>
          </tr>
        </thead>
        <tbody>
          {benchmark.runs.flatMap((run) =>
            METHOD_ORDER.map((key, i) => {
              const s = run.methods[key];
              return (
                <tr key={`${run.id}-${key}`} className={i === 0 ? "group-start" : undefined}>
                  <th scope="row">{i === 0 ? (run.ticker ?? benchmark.label) : ""}</th>
                  <td>
                    <span className="swatch" style={{ background: `var(--m-${key})` }} />
                    {METHOD_INFO[key].label}
                  </td>
                  <td className="num">
                    {pct(s.train_mean)} <small>± {pct(s.train_std)}</small>
                  </td>
                  <td className="num">
                    {pct(s.test_mean)} <small>± {pct(s.test_std)}</small>
                  </td>
                  <td className="num">{points(s.train_mean - s.test_mean)}</td>
                  <td className="num">{s.n_restarts ?? 10}</td>
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
    </div>
  );
}
