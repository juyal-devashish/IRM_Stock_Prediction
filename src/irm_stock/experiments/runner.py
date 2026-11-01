"""Shared multi-restart runner for the MLP-based experiments (finance, ColoredMNIST)."""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from irm_stock.methods import METHOD_LABELS, train_erm, train_irm, train_irm_games
from irm_stock.types import (
    Environment,
    IRMConfig,
    IRMGamesConfig,
    MethodSummary,
    MLPConfig,
    TrainResult,
)

logger = logging.getLogger(__name__)

#: Builds ``[train_env_0, train_env_1, test_env]`` for a given restart index. It may
#: draw from ``rng`` (the method's random stream) when the data itself is random.
EnvFactory = Callable[[int, np.random.RandomState], list[Environment]]


@dataclass
class MethodConfigs:
    erm: MLPConfig = field(default_factory=MLPConfig)
    irm: IRMConfig = field(default_factory=IRMConfig)
    irm_games: IRMGamesConfig = field(default_factory=IRMGamesConfig)


def train_method(method: str, envs: list[Environment], configs: MethodConfigs, rng) -> TrainResult:
    train_envs, test_env = envs[:-1], envs[-1]
    if method == "erm":
        return train_erm(train_envs, test_env, configs.erm, rng)
    if method == "irm":
        return train_irm(train_envs, test_env, configs.irm, rng)
    if method == "irm_games":
        return train_irm_games(train_envs, test_env, configs.irm_games, rng)
    raise ValueError(f"unknown method {method!r}")


def run_methods(
    env_factory: EnvFactory,
    methods: Sequence[str],
    configs: MethodConfigs,
    n_restarts: int,
    seed: int = 0,
) -> dict[str, MethodSummary]:
    """Train each method ``n_restarts`` times and collect final accuracies.

    Each method gets its own ``RandomState(seed)``, so a method's results do not
    depend on which other methods were run.
    """
    summaries: dict[str, MethodSummary] = {}
    for method in methods:
        logger.info("\nTraining %s", METHOD_LABELS[method])
        rng = np.random.RandomState(seed)
        summary = MethodSummary(METHOD_LABELS[method], [], [])
        for restart in range(n_restarts):
            logger.info("Restart %d", restart)
            envs = env_factory(restart, rng)
            result = train_method(method, envs, configs, rng)
            summary.train_accs.append(result.train_acc)
            summary.test_accs.append(result.test_acc)
            logger.info(
                "Final train acc (mean/std across restarts so far):\n%.5f %.5f\n"
                "Final test acc (mean/std across restarts so far):\n%.5f %.5f\n",
                summary.train_mean,
                summary.train_std,
                summary.test_mean,
                summary.test_std,
            )
        summaries[method] = summary
    return summaries


def format_summary(summaries: dict[str, MethodSummary]) -> str:
    lines = ["Summary (mean ± std):"]
    for s in summaries.values():
        lines.append(
            f"{s.method:10s} Train: {s.train_mean:.1%} ± {s.train_std:.1%} | "
            f"Test: {s.test_mean:.1%} ± {s.test_std:.1%}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------------ outputs


def make_run_dir(output_root: str | Path, experiment: str, slug: str) -> Path:
    """Create ``<output_root>/<experiment>/<timestamp>_<slug>/`` and return it."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug).strip("_")
    run_dir = Path(output_root) / experiment / f"{stamp}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
