"""ERM, IRMv1, and IRM-Games training loops for the NumPy MLP.

All three share the same contract: train on ``train_envs``, report accuracy on
``test_env`` (the out-of-distribution environment), and draw every random number
(weight init, minibatches) from the explicit ``rng`` so runs are reproducible.

Reported final accuracies are those measured at the last step *before* its
parameter update, matching the original research scripts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from irm_stock.core import (
    Adam,
    ensemble_logits,
    ensemble_member_nll_gradients,
    forward,
    init_mlp,
    irm_penalty,
    irm_penalty_gradients,
    mean_accuracy,
    mean_nll,
    nll_gradients,
)
from irm_stock.types import Environment, IRMConfig, IRMGamesConfig, MLPConfig, TrainResult

logger = logging.getLogger(__name__)


def _check(train_envs: Sequence[Environment], steps: int) -> None:
    if len(train_envs) < 1:
        raise ValueError("need at least one training environment")
    if steps < 1:
        raise ValueError("steps must be >= 1")


def _should_log(step: int, steps: int, log_every: int) -> bool:
    return step % log_every == 0 or step == steps - 1


def train_erm(
    train_envs: Sequence[Environment],
    test_env: Environment,
    config: MLPConfig,
    rng: np.random.RandomState,
    log_every: int = 100,
) -> TrainResult:
    """Empirical risk minimisation on the pooled training environments."""
    _check(train_envs, config.steps)
    X_train = np.vstack([X for X, _ in train_envs])
    y_train = np.hstack([y for _, y in train_envs])
    X_test, y_test = test_env

    W, V = init_mlp(X_train.shape[1], config.hidden_dim, rng)
    opt = Adam([W, V], lr=config.lr)
    history: list[dict[str, float]] = []

    logger.info("step         train nll    train acc    test acc")
    for step in range(config.steps):
        Z, H, F = forward(X_train, W, V)
        train_nll = mean_nll(F[:, 0], y_train)
        train_acc = mean_accuracy(F[:, 0], y_train)
        log_now = _should_log(step, config.steps, log_every)
        if log_now:
            test_acc = mean_accuracy(forward(X_test, W, V)[2][:, 0], y_test)

        dW, dV = nll_gradients(X_train, y_train, Z, H, F, V)
        opt.step([dW + config.l2 * 2 * W, dV + config.l2 * 2 * V], t=step + 1)

        if log_now:
            history.append(
                {"step": step, "train_nll": train_nll, "train_acc": train_acc, "test_acc": test_acc}
            )
            if step % log_every == 0:
                logger.info(f"{step:<12} {train_nll:<12.5f} {train_acc:<12.5f} {test_acc:<12.5f}")

    return TrainResult(float(train_acc), float(test_acc), history, W, V)


def train_irm(
    train_envs: Sequence[Environment],
    test_env: Environment,
    config: IRMConfig,
    rng: np.random.RandomState,
    log_every: int = 100,
) -> TrainResult:
    """IRMv1: per-environment risk plus an annealed invariance penalty."""
    _check(train_envs, config.steps)
    n_envs = len(train_envs)
    X_test, y_test = test_env

    W, V = init_mlp(train_envs[0][0].shape[1], config.hidden_dim, rng)
    opt = Adam([W, V], lr=config.lr)
    history: list[dict[str, float]] = []

    logger.info("step         train nll    train acc    train penalty test acc")
    for step in range(config.steps):
        pw = config.penalty_weight if step >= config.penalty_anneal_iters else 1.0
        nlls, accs, pens = [], [], []
        grad_W = np.zeros_like(W)
        grad_V = np.zeros_like(V)

        for X, y in train_envs:
            Z, H, F = forward(X, W, V)
            nlls.append(mean_nll(F[:, 0], y))
            accs.append(mean_accuracy(F[:, 0], y))
            pens.append(irm_penalty(F[:, 0], y))
            dW_nll, dV_nll = nll_gradients(X, y, Z, H, F, V)
            dW_pen, dV_pen = irm_penalty_gradients(X, y, Z, H, F, V)
            grad_W += dW_nll + pw * dW_pen
            grad_V += dV_nll + pw * dV_pen

        train_nll, train_acc, train_penalty = np.mean(nlls), np.mean(accs), np.mean(pens)
        log_now = _should_log(step, config.steps, log_every)
        if log_now:
            test_acc = mean_accuracy(forward(X_test, W, V)[2][:, 0], y_test)

        grad_W /= float(n_envs)
        grad_V /= float(n_envs)
        grad_W += config.l2 * 2 * W
        grad_V += config.l2 * 2 * V
        if pw > 1.0:
            # Rescale so the effective step size stays comparable once the penalty kicks in.
            grad_W /= pw
            grad_V /= pw
        opt.step([grad_W, grad_V], t=step + 1)

        if log_now:
            history.append(
                {
                    "step": step,
                    "train_nll": train_nll,
                    "train_acc": train_acc,
                    "train_penalty": train_penalty,
                    "test_acc": test_acc,
                }
            )
            if step % log_every == 0:
                logger.info(
                    f"{step:<12} {train_nll:<12.5f} {train_acc:<12.5f} "
                    f"{train_penalty:<13.5f} {test_acc:<12.5f}"
                )

    return TrainResult(float(train_acc), float(test_acc), history, W, V)


def train_irm_games(
    train_envs: Sequence[Environment],
    test_env: Environment,
    config: IRMGamesConfig,
    rng: np.random.RandomState,
    log_every: int = 50,
) -> TrainResult:
    """IRM-Games (Ahuja et al., 2020): one player per training environment.

    Each player owns an MLP; the prediction is the average of all players' logits.
    Players take turns doing a minibatch Adam step on their own environment's loss
    of the *ensemble* prediction. Optional warm start trains each player on its own
    environment (standard ERM) first.
    """
    _check(train_envs, config.steps)
    n_players = len(train_envs)
    all_envs = [*train_envs, test_env]
    l2 = config.l2

    W, V = init_mlp(train_envs[0][0].shape[1], config.hidden_dim, rng, n_models=n_players)
    opts = [Adam([W[e], V[e]], lr=config.lr) for e in range(n_players)]
    t_player = [0] * n_players

    def sample_batch(e: int) -> tuple[np.ndarray, np.ndarray]:
        X_e, y_e = train_envs[e]
        idx = rng.randint(0, X_e.shape[0], size=min(config.batch_size, X_e.shape[0]))
        return X_e[idx], y_e[idx]

    if config.warm_start:
        total_points = sum(X.shape[0] for X, _ in train_envs)
        for _ in range(max(1, total_points // config.batch_size)):
            for e in range(n_players):
                Xb, yb = sample_batch(e)
                Z, H, F = forward(Xb, W[e], V[e])
                dW, dV = nll_gradients(Xb, yb, Z, H, F, V[e])
                dW += 2 * l2 * W[e]
                dV += 2 * l2 * V[e]
                t_player[e] += 1
                opts[e].step([dW, dV], t=t_player[e])

    history: list[dict[str, float]] = []
    logger.info("step         train nll    train acc    train penalty test acc")
    for step in range(config.steps):
        for e in range(n_players):
            Xb, yb = sample_batch(e)
            logits_sum = np.zeros(Xb.shape[0], dtype=np.float32)
            for k in range(n_players):
                Z_k, H_k, F_k = forward(Xb, W[k], V[k])
                logits_sum += F_k[:, 0]
                if k == e:
                    Z_e, H_e = Z_k, H_k
            ens = logits_sum / float(n_players)

            dW, dV = ensemble_member_nll_gradients(Xb, yb, Z_e, H_e, ens, V[e], n_players)
            dW += 2 * l2 * W[e]
            dV += 2 * l2 * V[e]
            t_player[e] += 1
            opts[e].step([dW, dV], t=t_player[e])

        if _should_log(step, config.steps, log_every):
            nlls, accs, pens = [], [], []
            for X, y in all_envs:
                logits = ensemble_logits(X, W, V)
                nlls.append(mean_nll(logits, y))
                accs.append(mean_accuracy(logits, y))
                pens.append(irm_penalty(logits, y))
            train_nll = np.mean(nlls[:n_players])
            train_acc = np.mean(accs[:n_players])
            train_penalty = np.mean(pens[:n_players])
            test_acc = accs[n_players]
            history.append(
                {
                    "step": step,
                    "train_nll": train_nll,
                    "train_acc": train_acc,
                    "train_penalty": train_penalty,
                    "test_acc": test_acc,
                }
            )
            if step % log_every == 0:
                logger.info(
                    f"{step:<12} {train_nll:<12.5f} {train_acc:<12.5f} "
                    f"{train_penalty:<13.5f} {test_acc:<12.5f}"
                )

    return TrainResult(float(train_acc), float(test_acc), history, W, V)
