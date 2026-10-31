"""Minimal Adam optimiser that updates NumPy arrays in place."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Adam:
    """Adam over a fixed list of parameter arrays.

    Parameters are updated in place, so views (e.g. ``W[i]`` of a stacked ensemble)
    work as expected. The caller passes the bias-correction step ``t`` explicitly,
    because some methods (IRM-Games) keep per-player step counters.
    """

    def __init__(
        self,
        params: Sequence[np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def step(self, grads: Sequence[np.ndarray], t: int) -> None:
        """Apply one update. ``t`` is the 1-based step used for bias correction."""
        if len(grads) != len(self.params):
            raise ValueError(f"expected {len(self.params)} gradients, got {len(grads)}")
        b1, b2 = self.beta1, self.beta2
        for p, g, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            m[...] = b1 * m + (1 - b1) * g
            v[...] = b2 * v + (1 - b2) * (g**2)
            m_hat = m / (1 - b1**t)
            v_hat = v / (1 - b2**t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
