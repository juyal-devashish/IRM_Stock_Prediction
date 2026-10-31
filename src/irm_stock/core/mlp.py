"""One-hidden-layer ReLU MLP with a scalar logit output, implemented in NumPy.

Shapes used throughout:
    X: (n, d) float32 inputs
    W: (h, d) first-layer weights
    V: (1, h) output weights
    Z = X @ W.T  (pre-activation), H = relu(Z), F = H @ V.T  (logits, shape (n, 1))
"""

from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def forward(
    X: np.ndarray, W: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Z, H, F): pre-activations, hidden activations, and logits of shape (n, 1)."""
    Z = X @ W.T
    H = relu(Z)
    F = H @ V.T
    return Z, H, F


def init_mlp(
    input_dim: int,
    hidden_dim: int,
    rng: np.random.RandomState,
    n_models: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Glorot-uniform initialisation of (W, V).

    With ``n_models`` set, returns stacked parameters of shape (n_models, h, d) and
    (n_models, 1, h) for an ensemble (used by IRM-Games). W is drawn before V, which
    matters for reproducing the original random streams.
    """
    lead = () if n_models is None else (n_models,)
    limit_W = np.sqrt(6.0 / (input_dim + hidden_dim))
    W = rng.uniform(-limit_W, limit_W, (*lead, hidden_dim, input_dim)).astype(np.float32)
    limit_V = np.sqrt(6.0 / (hidden_dim + 1))
    V = rng.uniform(-limit_V, limit_V, (*lead, 1, hidden_dim)).astype(np.float32)
    return W, V


def ensemble_logits(X: np.ndarray, W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Average logits of an ensemble with stacked params W: (k, h, d), V: (k, 1, h)."""
    num_models = W.shape[0]
    logits_sum = np.zeros(X.shape[0], dtype=np.float32)
    for e in range(num_models):
        _, _, F = forward(X, W[e], V[e])
        logits_sum += F[:, 0]
    return logits_sum / float(num_models)


def predict_proba(X: np.ndarray, W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """P(y=1 | X) for a single model (W 2-D) or an ensemble (W 3-D)."""
    logits = ensemble_logits(X, W, V) if W.ndim == 3 else forward(X, W, V)[2][:, 0]
    return sigmoid(logits)
