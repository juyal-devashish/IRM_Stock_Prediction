"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: An environment is a (features, binary labels) pair: X of shape (n, d), y of shape (n,).
Environment = tuple[np.ndarray, np.ndarray]


@dataclass
class MLPConfig:
    """Hyper-parameters shared by the MLP-based methods (ERM, IRM, IRM-Games)."""

    hidden_dim: int = 256
    l2: float = 1e-3
    lr: float = 1e-3
    steps: int = 501


@dataclass
class IRMConfig(MLPConfig):
    penalty_weight: float = 10_000.0
    penalty_anneal_iters: int = 100


@dataclass
class IRMGamesConfig(MLPConfig):
    lr: float = 2.5e-4
    steps: int = 500
    batch_size: int = 256
    warm_start: bool = True


@dataclass
class TrainResult:
    """Outcome of one training run (one restart) of one method."""

    train_acc: float
    test_acc: float
    #: Logged metrics, one dict per logged step.
    history: list[dict[str, float]] = field(default_factory=list)
    #: Trained parameters: W, V (stacked along axis 0 for ensembles).
    W: np.ndarray | None = field(default=None, repr=False)
    V: np.ndarray | None = field(default=None, repr=False)


@dataclass
class MethodSummary:
    """Per-restart final accuracies of one method."""

    method: str
    train_accs: list[float]
    test_accs: list[float]

    @property
    def train_mean(self) -> float:
        return float(np.mean(self.train_accs))

    @property
    def train_std(self) -> float:
        return float(np.std(self.train_accs))

    @property
    def test_mean(self) -> float:
        return float(np.mean(self.test_accs))

    @property
    def test_std(self) -> float:
        return float(np.std(self.test_accs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "train_accs": [float(a) for a in self.train_accs],
            "test_accs": [float(a) for a in self.test_accs],
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "test_mean": self.test_mean,
            "test_std": self.test_std,
        }
