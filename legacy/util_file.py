import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(np.float32)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(X, W, V):
    Z = X @ W.T
    H = relu(Z)
    F = H @ V.T
    return Z, H, F

def mean_nll(logits, y):
    """Binary cross entropy with logits"""
    sig = sigmoid(logits)
    eps = 1e-7
    return -np.mean(y * np.log(sig + eps) + (1 - y) * np.log(1 - sig + eps))

def mean_accuracy(logits, y):
    preds = (logits > 0.).astype(np.float32)
    return np.mean(np.abs(preds - y) < 1e-2)


def penalty(logits, y):
    """IRM penalty"""
    sig = sigmoid(logits)
    grad_scale = np.mean(logits * (sig - y))
    return grad_scale ** 2

def compute_penalty_gradients(X, y, Z, H, F, W, V):
    B = X.shape[0]
    
    sig = sigmoid(F[:, 0])
    grad_scale = np.mean(F[:, 0] * (sig - y))
    d_grad_scale_dF = ((sig - y) + F[:, 0] * sig * (1 - sig)) / B
    dP_dF = 2 * grad_scale * d_grad_scale_dF
    dP_dF = dP_dF[:, None]
    
    dP_dV = dP_dF.T @ H
    dP_dH = dP_dF @ V
    dP_dZ = dP_dH * relu_grad(Z)
    dP_dW = dP_dZ.T @ X
    
    return dP_dW, dP_dV


def gradients_nll(X, y, Z, H, F, W, V):
    B = X.shape[0]
    
    sig = sigmoid(F[:, 0])
    dL_dF = (sig - y)[:, None]
    
    dL_dV = (dL_dF.T @ H) / B
    dL_dH = dL_dF @ V
    dL_dZ = dL_dH * relu_grad(Z)
    dL_dW = (dL_dZ.T @ X) / B
    
    return dL_dW, dL_dV

def gradients_nll_ensemble_env(X, y, Z_env, H_env, ensemble_logits, W_env, V_env, num_envs):
    B = X.shape[0]

    if ensemble_logits.ndim > 1: # Vector of logits must be flat
        g = ensemble_logits[:, 0]
    else:
        g = ensemble_logits

    sig = sigmoid(g)

    dL_dg = (sig - y)[:, None]
    dL_dF_env = (1.0/num_envs)*dL_dg

    dL_dV = (dL_dF_env.T @ H_env) / B
    dL_dH = dL_dF_env @ V_env
    dL_dZ = dL_dH * relu_grad(Z_env)
    dL_dW = (dL_dZ.T @ X) / B

    return dL_dW, dL_dV


def ensemble_logits_on_env(X_env, W_list, V_list):
    num_envs = W_list.shape[0]
    B = X_env.shape[0]
    logits_sum = np.zeros(B, dtype=np.float32)

    for e in range(num_envs):
        Z_e, H_e, F_e = forward(X_env, W_list[e], V_list[e])
        logits_sum += F_e[:, 0]
    return logits_sum / float(num_envs)
