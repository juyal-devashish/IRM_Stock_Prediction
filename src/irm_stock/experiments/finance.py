"""Finance experiments: can IRM / IRM-Games ignore a spurious feature in stock data?"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from irm_stock.data.finance import ENV_MODES, build_finance_envs, download_prices
from irm_stock.experiments.runner import (
    MethodConfigs,
    format_summary,
    make_run_dir,
    run_methods,
    write_json,
)
from irm_stock.methods import METHODS, normalize_method
from irm_stock.types import MethodSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Preset:
    name: str
    env_mode: str
    ticker: str
    label_noise: float


#: The three experiments reported in the project.
PRESETS: dict[int, Preset] = {
    1: Preset(
        "Experiment 1: time split (spurious changes over time)", "time_spurious", "AAPL", 0.0
    ),
    2: Preset(
        "Experiment 2: market regimes (bull / sideways / bear)", "trend_regimes", "SPY", 0.01
    ),
    3: Preset("Experiment 3: volatility regimes (low / mid / high)", "volatility", "WMT", 0.01),
}


@dataclass
class FinanceConfig:
    env_mode: str = "time_spurious"
    ticker: str = "AAPL"
    start: str = "2010-01-01"
    end: str = "2024-01-01"
    window: int = 10
    label_noise: float = 0.01
    n_restarts: int = 10
    seed: int = 0
    methods: Sequence[str] = METHODS
    models: MethodConfigs = field(default_factory=MethodConfigs)

    def __post_init__(self) -> None:
        if self.env_mode not in ENV_MODES:
            raise ValueError(f"unknown env_mode {self.env_mode!r}")
        self.ticker = self.ticker.upper()
        self.methods = tuple(normalize_method(m) for m in self.methods)

    @classmethod
    def from_preset(cls, experiment: int, **overrides: Any) -> FinanceConfig:
        p = PRESETS[experiment]
        base = {"env_mode": p.env_mode, "ticker": p.ticker, "label_noise": p.label_noise}
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)


@dataclass
class FinanceResult:
    config: FinanceConfig
    summaries: dict[str, MethodSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": "finance",
            "config": asdict(self.config),
            "results": {k: s.to_dict() for k, s in self.summaries.items()},
        }


def run_finance(config: FinanceConfig, prices: np.ndarray | None = None) -> FinanceResult:
    """Run every configured method. Pass ``prices`` to skip the download (e.g. in tests)."""
    if prices is None:
        prices = download_prices(config.ticker, config.start, config.end)

    def env_factory(restart: int, _rng: np.random.RandomState):
        return build_finance_envs(
            prices,
            window=config.window,
            env_mode=config.env_mode,
            seed=config.seed + restart,
            label_noise=config.label_noise,
        )

    summaries = run_methods(
        env_factory, config.methods, config.models, config.n_restarts, seed=config.seed
    )
    return FinanceResult(config, summaries)


def save_finance_result(result: FinanceResult, output_root: str | Path, tag: str = "") -> Path:
    """Write ``results.json`` and ``accuracy.png`` to a new run directory."""
    from irm_stock.plotting import plot_accuracy_and_gap

    cfg = result.config
    slug = "_".join(filter(None, [tag, cfg.env_mode, cfg.ticker.lower()]))
    run_dir = make_run_dir(output_root, "finance", slug)
    write_json(run_dir / "results.json", result.to_dict())
    plot_accuracy_and_gap(
        result.summaries, run_dir / "accuracy.png", title=f"{cfg.env_mode} — {cfg.ticker}"
    )
    logger.info("\n%s\n\nSaved results to %s", format_summary(result.summaries), run_dir)
    return run_dir
