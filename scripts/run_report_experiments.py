"""Run the three report experiments for extra tickers and save curated results.

Writes results/runs/finance/exp<N>_<TICKER>.json (committed), which
scripts/build_dashboard_data.py adds to the dashboard alongside the original logs.
Uses the package defaults (10 restarts, seed 0), so reruns are reproducible.

    uv run python scripts/run_report_experiments.py --tickers SPY,NVDA --experiments 1,2,3
    uv run python scripts/run_report_experiments.py --skip-existing   # resume
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import asdict
from pathlib import Path

from irm_stock.experiments.finance import FinanceConfig, run_finance

OUT = Path(__file__).resolve().parents[1] / "results" / "runs" / "finance"
DEFAULT_TICKERS = "SPY,AAPL,NVDA,MSFT,GOOGL,AMZN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", default=DEFAULT_TICKERS)
    parser.add_argument("--experiments", default="1,2,3")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    OUT.mkdir(parents=True, exist_ok=True)
    for exp in (int(e) for e in args.experiments.split(",")):
        for ticker in (t.strip().upper() for t in args.tickers.split(",") if t.strip()):
            path = OUT / f"exp{exp}_{ticker}.json"
            if args.skip_existing and path.exists():
                print(f"skip {path.name}")
                continue
            config = FinanceConfig.from_preset(exp, ticker=ticker)
            result = run_finance(config)
            payload = {
                "experiment": exp,
                "ticker": ticker,
                "computed_on": dt.date.today().isoformat(),
                "config": asdict(config),
                "results": {k: s.to_dict() for k, s in result.summaries.items()},
            }
            path.write_text(json.dumps(payload, indent=2) + "\n")
            line = ", ".join(f"{s.method} {s.test_mean:.1%}" for s in result.summaries.values())
            print(f"exp{exp} {ticker}: test {line}", flush=True)


if __name__ == "__main__":
    main()
