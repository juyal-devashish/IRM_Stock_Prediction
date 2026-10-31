"""ColoredMNIST benchmark: ERM vs IRM vs IRM-Games."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from irm_stock.data.colored_mnist import load_mnist, make_colored_mnist_envs
from irm_stock.experiments.runner import (
    MethodConfigs,
    format_summary,
    make_run_dir,
    run_methods,
    write_json,
)
from irm_stock.methods import METHODS, normalize_method
from irm_stock.types import IRMGamesConfig, MethodSummary

logger = logging.getLogger(__name__)


def _default_models() -> MethodConfigs:
    # IRM-Games on ColoredMNIST used the ERM learning rate and no warm start.
    return MethodConfigs(irm_games=IRMGamesConfig(lr=1e-3, steps=500, warm_start=False))


@dataclass
class ColoredMNISTConfig:
    n_restarts: int = 10
    seed: int = 0
    methods: Sequence[str] = METHODS
    data_dir: str | None = None
    models: MethodConfigs = field(default_factory=_default_models)

    def __post_init__(self) -> None:
        self.methods = tuple(normalize_method(m) for m in self.methods)


@dataclass
class ColoredMNISTResult:
    config: ColoredMNISTConfig
    summaries: dict[str, MethodSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": "colored_mnist",
            "config": asdict(self.config),
            "results": {k: s.to_dict() for k, s in self.summaries.items()},
        }


def run_colored_mnist(
    config: ColoredMNISTConfig, mnist: tuple[np.ndarray, np.ndarray] | None = None
) -> ColoredMNISTResult:
    """Run every configured method. Pass ``mnist=(images, labels)`` to skip loading."""
    images, labels = mnist if mnist is not None else load_mnist(config.data_dir)

    def env_factory(_restart: int, rng: np.random.RandomState):
        # Colouring and label noise are re-sampled every restart from the method's stream.
        return make_colored_mnist_envs(images, labels, rng)

    summaries = run_methods(
        env_factory, config.methods, config.models, config.n_restarts, seed=config.seed
    )
    return ColoredMNISTResult(config, summaries)


def save_colored_mnist_result(result: ColoredMNISTResult, output_root: str | Path) -> Path:
    from irm_stock.plotting import plot_train_test_bars

    run_dir = make_run_dir(output_root, "colored_mnist", "erm_irm_irmgames")
    write_json(run_dir / "results.json", result.to_dict())
    plot_train_test_bars(
        result.summaries,
        run_dir / "accuracy.png",
        title="ColoredMNIST: Train and Test Accuracy",
    )
    logger.info("\n%s\n\nSaved results to %s", format_summary(result.summaries), run_dir)
    return run_dir
