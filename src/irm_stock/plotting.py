"""Figures for experiment results.

Uses matplotlib's object-oriented API (no pyplot), so it is headless- and
thread-safe (e.g. inside the API server) and never opens a window.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from irm_stock.types import MethodSummary


def _bars(ax, summaries: Mapping[str, MethodSummary], capsize: float = 0) -> None:
    labels = [s.method for s in summaries.values()]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(
        x - width / 2,
        [s.train_mean for s in summaries.values()],
        width,
        yerr=[s.train_std for s in summaries.values()],
        label="Train",
        capsize=capsize,
    )
    ax.bar(
        x + width / 2,
        [s.test_mean for s in summaries.values()],
        width,
        yerr=[s.test_std for s in summaries.values()],
        label="Test",
        capsize=capsize,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def plot_accuracy_and_gap(
    summaries: Mapping[str, MethodSummary], path: str | Path, title: str | None = None
) -> Path:
    """Two panels: train/test accuracy per method, and the generalisation gap."""
    fig = Figure(figsize=(12, 4))
    ax1, ax2 = fig.subplots(1, 2)
    _bars(ax1, summaries)
    ax1.set_title("Accuracy")

    labels = [s.method for s in summaries.values()]
    ax2.bar(labels, [s.train_mean - s.test_mean for s in summaries.values()])
    ax2.set_ylabel("Train - Test Accuracy")
    ax2.set_title("Generalization Gap")
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return Path(path)


def plot_train_test_bars(
    summaries: Mapping[str, MethodSummary], path: str | Path, title: str
) -> Path:
    fig = Figure(figsize=(10, 5))
    ax = fig.subplots()
    _bars(ax, summaries, capsize=5)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return Path(path)


SYNTHETIC_COLORS = {"erm": "#5DA5DA", "irm_games": "#FAA43A", "irm": "#60BD68"}
SYNTHETIC_LABELS = {"erm": "ERM", "irm_games": "IRM Games", "irm": "IRM"}
SYNTHETIC_PANEL_ORDER = ["FOU", "FOS", "FEU", "FES", "POU", "POS", "PEU", "PES"]


def plot_synthetic_figure(
    results: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    path: str | Path,
    methods: Sequence[str] | None = None,
    palette: Mapping[str, str] | None = None,
    dpi: int = 300,
) -> Path:
    """Reproduce Figure 4 of the IRM paper: per config, causal (solid) and
    non-causal (hatched) weight errors for each method, log scale.

    ``methods`` picks and orders the methods drawn (default: all three);
    ``palette`` overrides their colours.
    """
    methods = list(methods or SYNTHETIC_COLORS)
    palette = {**SYNTHETIC_COLORS, **(palette or {})}
    colors = [palette[m] for m in methods]
    fig = Figure(figsize=(14, 6))
    axes = fig.subplots(2, 4).flatten()
    x = np.arange(len(methods))
    width = 0.35

    for idx, name in enumerate(SYNTHETIC_PANEL_ORDER):
        ax = axes[idx]
        if name not in results:
            ax.set_axis_off()
            continue
        res = results[name]
        for offset, key, extra in (
            (-width / 2, "causal_errors", {}),
            (
                width / 2,
                "noncausal_errors",
                {"hatch": "///", "edgecolor": "black", "linewidth": 0.5},
            ),
        ):
            ax.bar(
                x + offset,
                [np.mean(res[m][key]) for m in methods],
                width,
                yerr=[np.std(res[m][key]) for m in methods],
                color=colors,
                alpha=0.8,
                capsize=3,
                **extra,
            )
        ax.set_yscale("log")
        ax.set_xticks([])
        ax.set_title(name, fontsize=12, fontweight="bold")
        if idx % 4 == 0:
            ax.set_ylabel("causal error" if idx < 4 else "non-causal error", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    axes[-1].legend(
        handles=[Patch(facecolor=palette[m], label=SYNTHETIC_LABELS[m]) for m in methods],
        loc="center",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return Path(path)
