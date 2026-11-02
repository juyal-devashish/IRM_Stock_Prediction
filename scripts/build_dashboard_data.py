"""Build the dashboard's data from the curated files in results/.

Writes frontend/public/data/results.json and draws the method figures (ERM and IRM,
from the same numbers as the table) into frontend/public/figures/. Run from the repo root:

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


#: Methods drawn in figures, in the site's colours (IRM-Games is in the tables only).
CHART_METHODS = {"erm": ("ERM", "#2a78d6"), "irm": ("IRM", "#eb6834")}
INK, INK_2, GRID = "#16243a", "#4a5668", "#dfe4e1"
SYNTHETIC_RESULTS = ROOT / "results" / "runs" / "synthetic" / "figure4.json"

#: (benchmark id, run id, output file, caption): method figures drawn from the
#: same numbers as the dashboard's table, so they always agree with it.
METHOD_FIGURES = [
    (
        "market-regimes",
        "market-regimes-spy",
        "finance-trend-regimes-spy.png",
        "Market regimes on the S&P 500 (SPY): train and test accuracy, and the gap between them",
    ),
    (
        "volatility",
        "volatility-wmt",
        "finance-volatility-wmt.png",
        "Volatility regimes on WMT: train and test accuracy, and the gap between them",
    ),
    (
        "time-split",
        "time-split-aapl",
        "finance-time-split-aapl.png",
        "Time split on AAPL: train and test accuracy, and the gap between them",
    ),
    (
        "colored-mnist",
        "colored-mnist",
        "colored-mnist.png",
        "ColoredMNIST: train and test accuracy for ERM and IRM",
    ),
]


def draw_method_figure(methods: dict, path: Path) -> None:
    """Train vs test accuracy (with a coin-flip line) and the train-test gap."""
    from matplotlib.figure import Figure

    keys = list(CHART_METHODS)
    labels = [CHART_METHODS[k][0] for k in keys]
    colors = [CHART_METHODS[k][1] for k in keys]
    x = np.arange(len(keys))
    w = 0.36
    fig = Figure(figsize=(11, 4.2))
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(INK_2)
        ax.tick_params(colors=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    tr = [methods[k]["train_mean"] for k in keys]
    te = [methods[k]["test_mean"] for k in keys]
    err = {"ecolor": INK_2, "capsize": 3}
    ax1.bar(
        x - w / 2,
        tr,
        w,
        color=colors,
        alpha=0.38,
        yerr=[methods[k]["train_std"] for k in keys],
        **err,
    )
    ax1.bar(x + w / 2, te, w, color=colors, yerr=[methods[k]["test_std"] for k in keys], **err)
    for xi, k in enumerate(keys):
        m = methods[k]
        # Labels sit above each bar's error bar.
        y_tr = m["train_mean"] + m["train_std"] + 0.02
        y_te = m["test_mean"] + m["test_std"] + 0.02
        ax1.text(xi - w / 2, y_tr, f"{m['train_mean']:.0%}", ha="center", color=INK_2, fontsize=9)
        ax1.text(
            xi + w / 2,
            y_te,
            f"{m['test_mean']:.0%}",
            ha="center",
            color=INK,
            fontsize=9,
            weight="bold",
        )
    ax1.axhline(0.5, color=INK, linestyle=(0, (5, 4)), linewidth=1)
    ax1.text((len(keys) - 1) / 2, 0.515, "coin flip", ha="center", color=INK_2, fontsize=9)
    ax1.set_xticks(x, [f"{lab}\ntrain   test" for lab in labels], color=INK)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax1.set_title("Accuracy", color=INK, loc="left", fontsize=11)
    gaps = [a - b for a, b in zip(tr, te, strict=True)]
    ax2.bar(x, gaps, 0.55, color=colors)
    for xi, g in enumerate(gaps):
        pts = round(100 * g) or 0  # avoid "-0"
        ax2.text(xi, max(g, 0) + 0.015, f"{pts} pts", ha="center", color=INK, fontsize=9)
    ax2.axhline(0, color=INK_2, linewidth=0.8)
    ax2.set_xticks(x, labels, color=INK)
    ax2.set_title("Train minus test (points)", color=INK, loc="left", fontsize=11)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


def build_figures(benchmarks: list[dict]) -> list[dict]:
    from irm_stock.plotting import plot_synthetic_figure

    dest = OUT / "figures"
    shutil.rmtree(dest, ignore_errors=True)  # no stale images from earlier builds
    dest.mkdir(parents=True)
    runs = {r["id"]: r for b in benchmarks for r in b["runs"]}
    figures = []
    if SYNTHETIC_RESULTS.exists():
        results = json.loads(SYNTHETIC_RESULTS.read_text())["results"]
        plot_synthetic_figure(
            results,
            dest / "synthetic-figure4.png",
            methods=list(CHART_METHODS),
            palette={k: c for k, (_, c) in CHART_METHODS.items()},
            dpi=150,
        )
        figures.append(
            {
                "file": "figures/synthetic-figure4.png",
                "caption": "Synthetic causal-error benchmark: ERM vs IRM across eight settings",
            }
        )
    for _bid, run_id, name, caption in METHOD_FIGURES:
        draw_method_figure(runs[run_id]["methods"], dest / name)
        figures.append({"file": f"figures/{name}", "caption": caption})
    shutil.copyfile(
        FIGS / "colored_mnist" / "colored_mnist_visualization.png",
        dest / "colored-mnist-samples.png",
    )
    figures.append(
        {
            "file": "figures/colored-mnist-samples.png",
            "caption": "ColoredMNIST digits: colour tracks the label in training "
            "and flips at test time",
        }
    )
    return figures


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
    figures = build_figures(benchmarks)

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
