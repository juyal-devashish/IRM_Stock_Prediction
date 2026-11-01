"""Export parity fixtures for the TypeScript training lab (frontend/src/lab).

Runs the real Python core and training functions in float64 on a tiny toy problem
and records every random draw, so the TypeScript port can replay them exactly.
Regenerate after changing src/irm_stock/core or src/irm_stock/methods:

    uv run python scripts/export_lab_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from irm_stock.core import (
    Adam,
    ensemble_member_nll_gradients,
    forward,
    irm_penalty,
    irm_penalty_gradients,
    mean_accuracy,
    mean_nll,
    nll_gradients,
)
from irm_stock.methods import mlp_methods
from irm_stock.types import IRMConfig, IRMGamesConfig, MLPConfig

OUT = Path(__file__).resolve().parents[1] / "frontend" / "src" / "lab" / "parity.fixture.json"
H = 4


class RecordingRandomState(np.random.RandomState):
    """Records the uniform draws and integer indices the trainers consume."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.uniforms: list[float] = []
        self.ints: list[int] = []

    def uniform(self, low=0.0, high=1.0, size=None):  # type: ignore[override]
        u = self.random_sample(size)
        self.uniforms.extend(np.ravel(u).tolist())
        return low + (high - low) * u

    def randint(self, low, high=None, size=None, dtype=int):  # type: ignore[override]
        r = super().randint(low, high, size, dtype)
        self.ints.extend(np.ravel(r).tolist())
        return r


def init_mlp_float64(input_dim, hidden_dim, rng, n_models=None):
    """Same draw order as core.mlp.init_mlp, without the float32 cast."""
    lead = () if n_models is None else (n_models,)
    lw = np.sqrt(6.0 / (input_dim + hidden_dim))
    W = rng.uniform(-lw, lw, (*lead, hidden_dim, input_dim))
    lv = np.sqrt(6.0 / (hidden_dim + 1))
    V = rng.uniform(-lv, lv, (*lead, 1, hidden_dim))
    return W, V


def toy_env(rs: np.random.RandomState, n: int, agree: float):
    y = (rs.rand(n) < 0.5).astype(np.float64)
    sign = 2 * y - 1
    s_sign = np.where(rs.rand(n) < agree, sign, -sign)
    X = np.stack([sign + 1.5 * rs.randn(n), s_sign + 0.1 * rs.randn(n)], axis=1)
    return X, y


def flat(a: np.ndarray) -> list[float]:
    return np.ravel(a).astype(np.float64).tolist()


def env_json(env) -> dict:
    X, y = env
    return {"rows": X.shape[0], "cols": X.shape[1], "X": flat(X), "y": flat(y)}


def main() -> None:
    rs = np.random.RandomState(1)
    envs = [toy_env(rs, 40, 0.9), toy_env(rs, 40, 0.8), toy_env(rs, 40, 0.1)]

    # ---- unit functions
    X, y = envs[0]
    W = rs.randn(H, 2)
    V = rs.randn(1, H)
    W2, V2 = rs.randn(H, 2), rs.randn(1, H)
    Z, Hh, F = forward(X, W, V)
    dW, dV = nll_gradients(X, y, Z, Hh, F, V)
    pW, pV = irm_penalty_gradients(X, y, Z, Hh, F, V)
    ens = (F[:, 0] + forward(X, W2, V2)[2][:, 0]) / 2
    eW, eV = ensemble_member_nll_gradients(X, y, Z, Hh, ens, V, 2)
    p = rs.randn(5)
    g1, g2 = rs.randn(5), rs.randn(5)
    adam_p = p.copy()
    opt = Adam([adam_p], lr=0.05)
    opt.step([g1], t=1)
    opt.step([g2], t=2)
    unit = {
        "W": flat(W),
        "V": flat(V),
        "W2": flat(W2),
        "V2": flat(V2),
        "F": flat(F),
        "nll": float(mean_nll(F[:, 0], y)),
        "acc": float(mean_accuracy(F[:, 0], y)),
        "penalty": float(irm_penalty(F[:, 0], y)),
        "nll_dW": flat(dW),
        "nll_dV": flat(dV),
        "pen_dW": flat(pW),
        "pen_dV": flat(pV),
        "ens_dW": flat(eW),
        "ens_dV": flat(eV),
        "adam": {"p0": flat(p), "g1": flat(g1), "g2": flat(g2), "lr": 0.05, "p2": flat(adam_p)},
    }

    # ---- full training runs through the real Python trainers
    mlp_methods.init_mlp = init_mlp_float64  # keep everything in float64 for tight parity
    base = {"hidden_dim": H, "lr": 0.01, "l2": 1e-3}
    runs = {}
    cases = {
        "erm": (mlp_methods.train_erm, MLPConfig(steps=6, **base)),
        "irm": (
            mlp_methods.train_irm,
            IRMConfig(steps=6, penalty_weight=100.0, penalty_anneal_iters=3, **base),
        ),
        "irm_games": (
            mlp_methods.train_irm_games,
            IRMGamesConfig(steps=4, batch_size=16, warm_start=True, **base),
        ),
    }
    for name, (fn, cfg) in cases.items():
        rec = RecordingRandomState(7)
        res = fn(envs[:2], envs[2], cfg, rec, log_every=10_000)
        runs[name] = {
            "config": {k: getattr(cfg, k) for k in vars(cfg)},
            "uniforms": rec.uniforms,
            "ints": rec.ints,
            "W": flat(res.W),
            "V": flat(res.V),
            "test_logits": flat(
                mlp_methods.ensemble_logits(envs[2][0], res.W, res.V)
                if res.W.ndim == 3
                else forward(envs[2][0], res.W, res.V)[2][:, 0]
            ),
        }

    OUT.write_text(
        json.dumps({"envs": [env_json(e) for e in envs], "hidden": H, "unit": unit, "runs": runs})
        + "\n"
    )
    print(f"Wrote {OUT.relative_to(OUT.parents[3])}")


if __name__ == "__main__":
    main()
