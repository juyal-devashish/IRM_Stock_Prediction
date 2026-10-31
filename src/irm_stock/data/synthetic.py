"""Linear structural equation model from Figure 4 of the IRM paper (Arjovsky et al.).

``x -> y -> z``: the target ``y`` is caused by ``x`` and causes ``z``. Environments
scale the noise, so the regression weight on ``z`` is not invariant across
environments while the weight on ``x`` is.
"""

from __future__ import annotations

import numpy as np

RegressionEnv = tuple[np.ndarray, np.ndarray]

#: Noise scales of the three training environments.
ENV_SCALES = (0.2, 2.0, 5.0)


def config_name(ones: bool, hetero: bool, scramble: bool) -> str:
    """Three-letter config code used in the paper, e.g. ``FOU`` or ``PES``.

    P/F: identity ("partially") vs random ("fully") causal weights; E/O: heteroskedastic
    vs homoskedastic noise on y; S/U: scrambled vs unscrambled observed features.
    """
    return ("P" if ones else "F") + ("E" if hetero else "O") + ("S" if scramble else "U")


#: All eight configurations in paper order: (ones, hetero, scramble).
CONFIGURATIONS: list[tuple[bool, bool, bool]] = [
    (ones, hetero, scramble)
    for ones in (False, True)
    for hetero in (False, True)
    for scramble in (False, True)
]


class ChainEquationModel:
    def __init__(
        self,
        dim: int,
        rng: np.random.RandomState,
        ones: bool = True,
        scramble: bool = False,
        hetero: bool = True,
        hidden: bool = False,
    ) -> None:
        self.rng = rng
        self.hetero = hetero
        self.hidden = hidden
        self.dim = dim // 2
        k = self.dim

        if ones:
            self.wxy = np.eye(k, dtype=np.float32)
            self.wyz = np.eye(k, dtype=np.float32)
        else:
            self.wxy = rng.randn(k, k).astype(np.float32) / dim
            self.wyz = rng.randn(k, k).astype(np.float32) / dim

        if scramble:
            self.scramble = np.linalg.qr(rng.randn(dim, dim))[0].astype(np.float32)
        else:
            self.scramble = np.eye(dim, dtype=np.float32)

        if hidden:
            self.whx = rng.randn(k, k).astype(np.float32) / dim
            self.why = rng.randn(k, k).astype(np.float32) / dim
            self.whz = rng.randn(k, k).astype(np.float32) / dim
        else:
            self.whx = np.eye(k, dtype=np.float32)
            self.why = np.zeros((k, k), dtype=np.float32)
            self.whz = np.zeros((k, k), dtype=np.float32)

    def solution(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (true invariant weights on [x, z], scramble matrix)."""
        w = np.concatenate([self.wxy.sum(1), np.zeros(self.dim)]).reshape(-1, 1)
        return w.astype(np.float32), self.scramble

    def sample(self, n: int, env: float) -> RegressionEnv:
        rng, k = self.rng, self.dim

        def noise() -> np.ndarray:
            return rng.randn(n, k).astype(np.float32)

        h = noise() * env
        x = h @ self.whx + noise() * env
        if self.hetero:
            y = x @ self.wxy + h @ self.why + noise() * env
            z = y @ self.wyz + h @ self.whz + noise()
        else:
            y = x @ self.wxy + h @ self.why + noise()
            z = y @ self.wyz + h @ self.whz + noise() * env

        X = np.concatenate([x, z], axis=1) @ self.scramble
        return X, y.sum(1, keepdims=True)

    __call__ = sample


def weight_errors(w_true: np.ndarray, w_hat: np.ndarray) -> tuple[float, float]:
    """Mean squared error on causal (non-zero) and non-causal (zero) true weights."""
    w_true, w_hat = w_true.flatten(), w_hat.flatten()
    causal = w_true != 0
    err_c = np.mean((w_true[causal] - w_hat[causal]) ** 2) if causal.any() else 0.0
    err_nc = np.mean((w_true[~causal] - w_hat[~causal]) ** 2) if (~causal).any() else 0.0
    return float(err_c), float(err_nc)
