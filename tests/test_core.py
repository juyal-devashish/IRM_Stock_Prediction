import numpy as np
import pytest

from irm_stock.core import (
    Adam,
    forward,
    init_mlp,
    irm_penalty,
    irm_penalty_gradients,
    mean_nll,
    nll_gradients,
)


def _setup(seed=0, n=12, d=5, h=4):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (rng.rand(n) > 0.5).astype(np.float64)
    W, V = init_mlp(d, h, rng)
    return X, y, W.astype(np.float64), V.astype(np.float64)


def _numeric_grad(f, P, eps=1e-6):
    g = np.zeros_like(P)
    for idx in np.ndindex(*P.shape):
        old = P[idx]
        P[idx] = old + eps
        hi = f()
        P[idx] = old - eps
        lo = f()
        P[idx] = old
        g[idx] = (hi - lo) / (2 * eps)
    return g


@pytest.mark.parametrize("which", ["nll", "penalty"])
def test_gradients_match_finite_differences(which):
    X, y, W, V = _setup()

    def loss():
        F = forward(X, W, V)[2][:, 0]
        return mean_nll(F, y) if which == "nll" else irm_penalty(F, y)

    Z, H, F = forward(X, W, V)
    grad_fn = nll_gradients if which == "nll" else irm_penalty_gradients
    dW, dV = grad_fn(X, y, Z, H, F, V)
    np.testing.assert_allclose(dW, _numeric_grad(loss, W), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(dV, _numeric_grad(loss, V), rtol=1e-3, atol=1e-6)


def test_adam_updates_in_place_and_rejects_bad_grads():
    p = np.ones(3, dtype=np.float32)
    view = p[:]
    opt = Adam([p], lr=0.1)
    opt.step([np.ones(3, dtype=np.float32)], t=1)
    assert np.all(view < 1.0)
    with pytest.raises(ValueError):
        opt.step([], t=2)


def test_init_mlp_ensemble_shapes():
    W, V = init_mlp(7, 5, np.random.RandomState(0), n_models=3)
    assert W.shape == (3, 5, 7) and V.shape == (3, 1, 5)
