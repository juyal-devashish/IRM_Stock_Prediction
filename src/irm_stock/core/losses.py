"""Binary cross-entropy, the IRMv1 penalty, and their analytic gradients for the MLP."""

from __future__ import annotations

import numpy as np

from irm_stock.core.mlp import relu_grad, sigmoid

_LOG_EPS = 1e-7


def mean_nll(logits: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy computed from logits."""
    sig = sigmoid(logits)
    return -np.mean(y * np.log(sig + _LOG_EPS) + (1 - y) * np.log(1 - sig + _LOG_EPS))


def mean_accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    preds = (logits > 0.0).astype(np.float32)
    return np.mean(np.abs(preds - y) < 1e-2)


def irm_penalty(logits: np.ndarray, y: np.ndarray) -> float:
    """IRMv1 penalty: squared gradient of the risk w.r.t. a dummy scalar classifier w=1."""
    sig = sigmoid(logits)
    grad_scale = np.mean(logits * (sig - y))
    return grad_scale**2


def nll_gradients(
    X: np.ndarray, y: np.ndarray, Z: np.ndarray, H: np.ndarray, F: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of :func:`mean_nll` w.r.t. (W, V)."""
    B = X.shape[0]
    sig = sigmoid(F[:, 0])
    dL_dF = (sig - y)[:, None]
    dL_dV = (dL_dF.T @ H) / B
    dL_dH = dL_dF @ V
    dL_dZ = dL_dH * relu_grad(Z)
    dL_dW = (dL_dZ.T @ X) / B
    return dL_dW, dL_dV


def irm_penalty_gradients(
    X: np.ndarray, y: np.ndarray, Z: np.ndarray, H: np.ndarray, F: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of :func:`irm_penalty` w.r.t. (W, V)."""
    B = X.shape[0]
    sig = sigmoid(F[:, 0])
    grad_scale = np.mean(F[:, 0] * (sig - y))
    d_grad_scale_dF = ((sig - y) + F[:, 0] * sig * (1 - sig)) / B
    dP_dF = (2 * grad_scale * d_grad_scale_dF)[:, None]
    dP_dV = dP_dF.T @ H
    dP_dH = dP_dF @ V
    dP_dZ = dP_dH * relu_grad(Z)
    dP_dW = dP_dZ.T @ X
    return dP_dW, dP_dV


def ensemble_member_nll_gradients(
    X: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    ensemble_logits: np.ndarray,
    V: np.ndarray,
    num_models: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of the ensemble's NLL w.r.t. one member's (W, V).

    The ensemble logit is the mean of member logits, so each member receives
    ``1 / num_models`` of the gradient flowing into the ensemble output.
    """
    B = X.shape[0]
    g = ensemble_logits[:, 0] if ensemble_logits.ndim > 1 else ensemble_logits
    sig = sigmoid(g)
    dL_dF = (1.0 / num_models) * (sig - y)[:, None]
    dL_dV = (dL_dF.T @ H) / B
    dL_dH = dL_dF @ V
    dL_dZ = dL_dH * relu_grad(Z)
    dL_dW = (dL_dZ.T @ X) / B
    return dL_dW, dL_dV
