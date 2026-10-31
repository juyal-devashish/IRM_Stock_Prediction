"""Figure 4 of the IRM paper on a linear SEM, with ICP replaced by IRM-Games.

The full run (8 configs x 10 reps x 3 methods x 100k full-batch Adam steps) takes
hours; use ``n_iterations`` / ``n_reps`` / ``configs`` to run a subset.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from irm_stock.data.synthetic import (
    CONFIGURATIONS,
    ENV_SCALES,
    ChainEquationModel,
    config_name,
    weight_errors,
)
from irm_stock.experiments.runner import make_run_dir, write_json
from irm_stock.methods.linear import fit_erm, fit_irm, fit_irm_games

logger = logging.getLogger(__name__)

#: Methods in the order they consume the random stream (keep for reproducibility).
SYNTHETIC_METHODS = {"erm": fit_erm, "irm_games": fit_irm_games, "irm": fit_irm}

ALL_CONFIG_NAMES = [config_name(*c) for c in CONFIGURATIONS]

#: {config name: {method: {"causal_errors": [...], "noncausal_errors": [...]}}}
SyntheticResults = dict[str, dict[str, dict[str, list[float]]]]


@dataclass
class SyntheticConfig:
    configs: Sequence[str] = field(default_factory=lambda: list(ALL_CONFIG_NAMES))
    n_reps: int = 10
    n_iterations: int = 100_000
    n_samples: int = 1000
    dim: int = 10
    lr: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        self.configs = [c.upper() for c in self.configs]
        unknown = set(self.configs) - set(ALL_CONFIG_NAMES)
        if unknown:
            raise ValueError(f"unknown configs {sorted(unknown)}; expected {ALL_CONFIG_NAMES}")


def run_configuration(
    ones: bool, hetero: bool, scramble: bool, config: SyntheticConfig
) -> dict[str, dict[str, list[float]]]:
    rng = np.random.RandomState(config.seed)
    results = {m: {"causal_errors": [], "noncausal_errors": []} for m in SYNTHETIC_METHODS}
    for _ in range(config.n_reps):
        sem = ChainEquationModel(config.dim, rng, ones=ones, scramble=scramble, hetero=hetero)
        envs = [sem.sample(config.n_samples, e) for e in ENV_SCALES]
        w_true, scramble_mat = sem.solution()
        for method, fit in SYNTHETIC_METHODS.items():
            w_hat = scramble_mat @ fit(envs, rng, lr=config.lr, n_iterations=config.n_iterations)
            err_c, err_nc = weight_errors(w_true, w_hat)
            results[method]["causal_errors"].append(err_c)
            results[method]["noncausal_errors"].append(err_nc)
    return results


def run_synthetic(config: SyntheticConfig) -> SyntheticResults:
    all_results: SyntheticResults = {}
    for ones, hetero, scramble in CONFIGURATIONS:
        name = config_name(ones, hetero, scramble)
        if name not in config.configs:
            continue
        logger.info("Running %s: ones=%s, hetero=%s, scramble=%s", name, ones, hetero, scramble)
        res = run_configuration(ones, hetero, scramble, config)
        all_results[name] = res
        for method, errs in res.items():
            c, nc = np.array(errs["causal_errors"]), np.array(errs["noncausal_errors"])
            logger.info(
                "  %-10s causal=%.5f±%.5f, non-causal=%.5f±%.5f",
                method,
                c.mean(),
                c.std(),
                nc.mean(),
                nc.std(),
            )
    return all_results


def save_synthetic_result(
    results: SyntheticResults, config: SyntheticConfig, output_root: str | Path
) -> Path:
    from irm_stock.plotting import plot_synthetic_figure

    run_dir = make_run_dir(output_root, "synthetic", f"{config.n_iterations}iters")
    payload: dict[str, Any] = {"experiment": "synthetic", "config": asdict(config)}
    payload["results"] = results
    write_json(run_dir / "results.json", payload)
    plot_synthetic_figure(results, run_dir / "figure4_irm_games.pdf")
    logger.info("Saved results to %s", run_dir)
    return run_dir
