"""Linear-regression versions of ERM, IRMv1 and IRM-Games (for the synthetic SEM task).

These fit ``y ≈ X @ w`` with full-batch Adam on squared error and return the
weight vector ``w`` of shape (d, 1).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from irm_stock.core.optim import Adam

RegressionEnv = tuple[np.ndarray, np.ndarray]  # X: (n, d), y: (n, 1)


def mse_gradient(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (2.0 / X.shape[0]) * X.T @ (X @ w - y)


def irm_penalty_gradient(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of ``||grad_w MSE||^2`` (IRMv1 penalty for a linear model)."""
    B = X.shape[0]
    grad_loss = (2.0 / B) * X.T @ (X @ w - y)
    hessian = (2.0 / B) * X.T @ X
    return 2.0 * hessian @ grad_loss


def _init_w(d: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.randn(d, 1).astype(np.float32) * 0.01


def fit_erm(
    envs: Sequence[RegressionEnv], rng: np.random.RandomState, lr: float, n_iterations: int
) -> np.ndarray:
    X = np.vstack([X for X, _ in envs])
    y = np.vstack([y for _, y in envs])
    w = _init_w(X.shape[1], rng)
    opt = Adam([w], lr=lr)
    for step in range(n_iterations):
        opt.step([mse_gradient(X, w, y)], t=step + 1)
    return w


def fit_irm(
    envs: Sequence[RegressionEnv],
    rng: np.random.RandomState,
    lr: float,
    n_iterations: int,
    penalty_weight: float = 10_000.0,
    penalty_anneal_iters: int | None = None,
) -> np.ndarray:
    """IRMv1. The penalty is switched on after ``penalty_anneal_iters`` (default 10%)."""
    if penalty_anneal_iters is None:
        penalty_anneal_iters = n_iterations // 10
    w = _init_w(envs[0][0].shape[1], rng)
    opt = Adam([w], lr=lr)
    for step in range(n_iterations):
        pw = penalty_weight if step >= penalty_anneal_iters else 1.0
        grad = np.zeros_like(w)
        for X, y in envs:
            grad += mse_gradient(X, w, y) + pw * irm_penalty_gradient(X, w, y)
        grad /= len(envs)
        if pw > 1.0:
            grad /= pw
        opt.step([grad], t=step + 1)
    return w


def fit_irm_games(
    envs: Sequence[RegressionEnv], rng: np.random.RandomState, lr: float, n_iterations: int
) -> np.ndarray:
    """IRM-Games with one linear player per environment, updated round-robin.

    Returns the averaged weights (equivalent to the ensemble for a linear model).

    Note: Adam bias correction uses the *global* iteration count rather than each
    player's own update count. This mirrors the original implementation that
    produced the published figures; do not "fix" it without re-running them.
    """
    n_players = len(envs)
    d = envs[0][0].shape[1]
    W = [_init_w(d, rng) for _ in range(n_players)]
    opts = [Adam([w], lr=lr) for w in W]
    for step in range(n_iterations):
        e = step % n_players
        X, y = envs[e]
        ens_pred = np.zeros((X.shape[0], 1), dtype=np.float32)
        for w in W:
            ens_pred += X @ w
        ens_pred /= n_players
        grad = (1.0 / n_players) * (2.0 / X.shape[0]) * X.T @ (ens_pred - y)
        opts[e].step([grad], t=step + 1)
    return np.mean(W, axis=0)
