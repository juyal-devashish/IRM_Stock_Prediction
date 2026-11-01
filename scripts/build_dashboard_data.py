"""Build the dashboard's data from the curated files in results/.

Writes frontend/public/data/results.json and copies selected figures to
frontend/public/figures/. Run from the repo root:

    uv run python scripts/build_dashboard_data.py

Only logs whose filename names the ticker are used; results/runs/finance/*.json
(package runs) fill in tickers that have no log. Unlabelled logs
(experimentN_output*.txt) and the truncated experiment_2_aapl_new.txt are skipped
because their ticker / completeness cannot be verified.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs" / "finance"
RUNS = ROOT / "results" / "runs" / "finance"  # from scripts/run_report_experiments.py
FIGS = ROOT / "results" / "figures"
OUT = ROOT / "frontend" / "public"

METHODS = {"ERM": "erm", "IRM": "irm", "IRM Games": "irm_games"}

# (benchmark id, label, env mode, one-line description)
FINANCE_BENCHMARKS = {
    1: (
        "time-split",
        "Time split",
        "time_spurious",
        "Price history in three time slices; the shortcut feature agrees with the label "
        "80% and 90% of the time in training and only 10% of the time in the test slice.",
    ),
    2: (
        "market-regimes",
        "Market regimes",
        "trend_regimes",
        "Bull and sideways markets for training, bear market for testing.",
    ),
    3: (
        "volatility",
        "Volatility",
        "volatility",
        "Low and mid volatility for training, high volatility for testing.",
    ),
}

# (experiment, ticker, file)
FINANCE_RUNS = [
    (1, "AAPL", "experiment_1_aapl_new.txt"),
    (1, "VOO", "experiment1_voo_output.txt"),
    (1, "WMT", "experiment1_wmt_output.txt"),
    (2, "VOO", "experiment2_voo_output.txt"),
    (2, "WMT", "experiment2_wmt_output.txt"),
    (3, "VOO", "experiment3_voo_output.txt"),
    (3, "WMT", "experiment3_wmt_output.txt"),
]

#: Display order of tickers in each benchmark (S&P 500 first, then by market cap).
TICKER_ORDER = ["SPY", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "VOO", "WMT"]
TICKER_LABELS = {"SPY": "S&P 500 (SPY)"}


def json_runs(exp: int, have: set[str]) -> list[dict]:
    """Curated package runs for tickers that have no original report log."""
    runs = []
    for path in sorted(RUNS.glob(f"exp{exp}_*.json")):
        data = json.loads(path.read_text())
        if data["ticker"] in have:
            continue
        methods = {
            k: {
                "train_mean": r["train_mean"],
                "train_std": r["train_std"],
                "test_mean": r["test_mean"],
                "test_std": r["test_std"],
                "n_restarts": len(r["test_accs"]),
            }
            for k, r in data["results"].items()
        }
        runs.append(
            {
                "ticker": data["ticker"],
                "source": f"results/runs/finance/{path.name}",
                "computed_on": data["computed_on"],
                "methods": methods,
            }
        )
    return runs


# Matches e.g. "ERM Results:\n  Train: 77.7% ± 0.4%\n  Test:  21.4% ± 0.6%" (± may be mojibake).
BLOCK = re.compile(
    r"^(ERM|IRM Games|IRM) Results:\s*\n\s*Train:\s*([\d.]+)%\s*\S+\s*([\d.]+)%\s*\n"
    r"\s*Test:\s*([\d.]+)%\s*\S+\s*([\d.]+)%",
    re.M,
)


def parse_log(path: Path) -> dict[str, dict[str, float]]:
    """Last summary block per method (logs may contain earlier partial runs)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    found: dict[str, dict[str, float]] = {}
    for name, tr, tr_sd, te, te_sd in BLOCK.findall(text):
        found[METHODS[name]] = {
            "train_mean": float(tr) / 100,
            "train_std": float(tr_sd) / 100,
            "test_mean": float(te) / 100,
            "test_std": float(te_sd) / 100,
        }
    missing = set(METHODS.values()) - set(found)
    if missing:
        raise ValueError(f"{path.name}: missing summary for {sorted(missing)}")
    return found


def colored_mnist_run() -> dict:
    data = np.load(ROOT / "results" / "data" / "results_cm.npz")
    methods = {}
    for key, short in (("erm", "erm"), ("irm", "irm"), ("irmg", "irm_games")):
        tr, te = data[f"{key}_train"], data[f"{key}_test"]
        methods[short] = {
            "train_mean": float(tr.mean()),
            "train_std": float(tr.std()),
            "test_mean": float(te.mean()),
            "test_std": float(te.std()),
            "n_restarts": len(tr),
        }
    return {
        "id": "colored-mnist",
        "ticker": None,
        "source": "results/data/results_cm.npz",
        "methods": methods,
    }


FIGURES = [
    (
        "synthetic/figure4_irm_games.png",
        "synthetic-figure4.png",
        "Synthetic causal-error benchmark: ERM vs IRM Games vs IRM across eight settings",
    ),
    (
        "finance/combined_finance_2_trend_regimes_spy.png",
        "finance-trend-regimes-spy.png",
        "Market regimes on SPY: train and test accuracy, and the gap between them",
    ),
    (
        "finance/combined_finance_3_volatility_wmt.png",
        "finance-volatility-wmt.png",
        "Volatility regimes on WMT: train and test accuracy, and the gap between them. "
        "From a later run than the WMT row in the table, so the numbers differ, "
        "most for IRM-Games",
    ),
    (
        "finance/combined_finance_1_time_spurious_aapl.png",
        "finance-time-split-aapl.png",
        "Time split on AAPL: train and test accuracy, and the gap between them. "
        "From a later run than the AAPL row in the table, so the numbers differ, "
        "most for IRM-Games",
    ),
    (
        "colored_mnist/erm_irm_irmgames_comparison.png",
        "colored-mnist.png",
        "ColoredMNIST: train and test accuracy for ERM, IRM and IRM-Games",
    ),
    (
        "colored_mnist/colored_mnist_visualization.png",
        "colored-mnist-samples.png",
        "ColoredMNIST digits: colour tracks the label in training and flips at test time",
    ),
]


def main(live: Path | None = None) -> None:
    benchmarks = [
        {
            "id": "colored-mnist",
            "label": "ColoredMNIST",
            "description": "Handwritten digits where colour agrees with the label 80-90% of "
            "the time in training and only 10% of the time at test.",
            "runs": [colored_mnist_run()],
        }
    ]
    for exp, (bid, label, _mode, desc) in FINANCE_BENCHMARKS.items():
        runs = [
            {
                "ticker": ticker,
                "source": f"results/logs/finance/{fname}",
                "methods": parse_log(LOGS / fname),
            }
            for e, ticker, fname in FINANCE_RUNS
            if e == exp
        ]
        runs += json_runs(exp, {r["ticker"] for r in runs})
        rank = {t: i for i, t in enumerate(TICKER_ORDER)}
        runs.sort(key=lambda r: rank.get(r["ticker"], len(rank)))
        for r in runs:
            r["id"] = f"{bid}-{r['ticker'].lower()}"
            r["label"] = TICKER_LABELS.get(r["ticker"], r["ticker"])
        benchmarks.append({"id": bid, "label": label, "description": desc, "runs": runs})

    OUT.joinpath("data").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    figures = []
    for src, dest, caption in FIGURES:
        shutil.copyfile(FIGS / src, OUT / "figures" / dest)
        figures.append({"file": f"figures/{dest}", "caption": caption})

    payload = {"benchmarks": benchmarks, "figures": figures}
    (OUT / "data" / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    if live is not None:
        # For local development: copy a walk-forward history (CI fetches it from the data branch).
        shutil.copyfile(live, OUT / "data" / "live.json")
    n = sum(len(b["runs"]) for b in benchmarks)
    print(f"Wrote {n} runs and {len(figures)} figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", type=Path, help="Walk-forward history.json to include.")
    main(parser.parse_args().live)
