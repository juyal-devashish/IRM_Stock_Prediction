from util_file import *

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# Download prices via yfinance
def download_prices(ticker, start="2010-01-01", end="2024-01-01"):
    data = yf.download(ticker, start=start, end=end, group_by="column")

    if data is None or len(data) == 0:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns if necessary
    if isinstance(data.columns, pd.MultiIndex):
        flat_cols = []
        for col in data.columns.values:
            flat = " ".join(str(c) for c in col if c not in ("", None))
            flat_cols.append(flat)
        data.columns = flat_cols

    cols = list(data.columns)
    target_col = None
    for cand in ["Adj Close", "Close"]:
        for c in cols:
            if cand in c:
                target_col = c
                break
        if target_col is not None:
            break

    if target_col is None:
        target_col = cols[-1]

    prices = data[target_col].values.astype(np.float32)
    return prices


def make_finance_environment(prices, window=10, e=0.2, seed=None):
    """
    Original environment:
      - Features: window of past returns
      - Label: sign of future return
      - Spurious feature s: correlated with (noisy) label at strength (1 - e)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    returns = prices[1:] / prices[:-1] - 1

    future_returns = returns[1:]
    labels = (future_returns > 0).astype(np.float32)

    X_list = []
    for t in range(window, len(returns) - 1):
        feat = returns[t-window:t]
        X_list.append(feat)
    X = np.stack(X_list, axis=0).astype(np.float32)

    y = labels[window:]

    # label noise
    flip = (rng.rand(len(y)) < 0.01).astype(np.float32)
    y_noisy = np.abs(y - flip)

    # spurious feature
    flip_spurious = (rng.rand(len(y_noisy)) < e).astype(np.float32)
    s = np.abs(y_noisy - flip_spurious)

    s = s[:, None]
    X_full = np.concatenate([X, s], axis=1)

    return X_full, y_noisy


def make_finance_envs_spurious_strength(
    prices,
    window=10,
    e_list=(0.1, 0.2, 0.9),
    flip_prob_list=None,
    seed=None,
):
    """
    Colored-MNIST style:
      - Single long price series
      - Multiple envs with different spurious strength e between s and label
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    if flip_prob_list is None:
        flip_prob_list = [0.01] * len(e_list)
    assert len(e_list) == len(flip_prob_list)

    returns = prices[1:] / prices[:-1] - 1
    future_returns = returns[1:]
    labels = (future_returns > 0).astype(np.float32)

    X_list = []
    for t in range(window, len(returns) - 1):
        feat = returns[t-window:t]
        X_list.append(feat)
    X_base = np.stack(X_list, axis=0).astype(np.float32)
    y_base = labels[window:]

    envs = []
    for e, flip_prob in zip(e_list, flip_prob_list):
        # label noise
        flip = (rng.rand(len(y_base)) < flip_prob).astype(np.float32)
        y_noisy = np.abs(y_base - flip)

        # spurious color: equals y_noisy with prob 1-e
        flip_spurious = (rng.rand(len(y_noisy)) < e).astype(np.float32)
        s = np.abs(y_noisy - flip_spurious)[:, None]

        X_full = np.concatenate([X_base, s], axis=1)
        envs.append((X_full, y_noisy))

    return envs

def make_factor_envs(prices, window=10, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    returns = prices[1:] / prices[:-1] - 1

    f1_list, f2_list = [], []
    for t in range(window, len(returns)):
        win = returns[t-window:t]
        f1_list.append(win.mean())
        f2_list.append(win.std() + 1e-6)
    f1 = np.array(f1_list)
    f2 = np.array(f2_list)

    # label uses ONLY f1: sign of future return
    future_returns = returns[window:]
    y = (future_returns > 0).astype(np.float32)

    n = min(len(f1), len(y))
    f1, f2, y = f1[:n], f2[:n], y[:n]

    def make_env(alpha, beta):
        x1 = f1  # invariant
        x2 = alpha * f1 + beta * f2
        X = np.stack([x1, x2], axis=1).astype(np.float32)
        return X, y

    # Env1, Env2: x2 correlated with y
    env1 = make_env(alpha=1.0, beta=0.5)
    env2 = make_env(alpha=0.8, beta=-0.7)
    # Env3: x2 anti-correlated with y → OOD env
    env3 = make_env(alpha=-1.0, beta=0.5)

    return [env1, env2, env3]

def make_trend_regime_envs(
    prices,
    window=10,
    short_window=50,
    long_window=200,
    side_eps=0.5,
    e_bull=0.2,
    e_side=0.2,
    e_bear=0.9,
    seed=None,
):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    prices_series = pd.Series(prices)
    ma_short = prices_series.rolling(short_window).mean()
    ma_long = prices_series.rolling(long_window).mean()
    trend = (ma_short - ma_long).values  # length = len(prices)

    # Base returns & labels (same pattern as original env)
    returns = prices[1:] / prices[:-1] - 1  # length N-1
    future_returns = returns[1:]            # length N-2
    labels = (future_returns > 0).astype(np.float32)

    # Threshold for "sideways": fraction of trend std
    valid_trend = trend[~np.isnan(trend)]
    if len(valid_trend) == 0:
        raise ValueError("Trend has too many NaNs; check short/long windows or date range.")
    std_trend = np.std(valid_trend)
    side_thresh = side_eps * std_trend  # |trend| <= side_thresh → sideways

    X_bull_list, y_bull_list = [], []
    X_side_list, y_side_list = [], []
    X_bear_list, y_bear_list = [], []

    # t indexes returns; we align trend with price index t+1
    for t in range(window, len(returns) - 1):
        idx_price = t + 1
        if idx_price >= len(trend):
            break

        tr = trend[idx_price]
        if np.isnan(tr):
            continue

        feat = returns[t-window:t]
        y_t = labels[t]  # consistent with original y = labels[window:]

        if tr > side_thresh:
            # Bull
            X_bull_list.append(feat)
            y_bull_list.append(y_t)
        elif tr < -side_thresh:
            # Bear
            X_bear_list.append(feat)
            y_bear_list.append(y_t)
        else:
            # Sideways
            X_side_list.append(feat)
            y_side_list.append(y_t)

    if len(X_bull_list) == 0 or len(X_side_list) == 0 or len(X_bear_list) == 0:
        raise ValueError(
            "Not enough samples in bull/sideways/bear regimes; "
            "try different dates, short/long windows, or side_eps."
        )

    X_bull = np.stack(X_bull_list, axis=0).astype(np.float32)
    y_bull = np.array(y_bull_list, dtype=np.float32)

    X_side = np.stack(X_side_list, axis=0).astype(np.float32)
    y_side = np.array(y_side_list, dtype=np.float32)

    X_bear = np.stack(X_bear_list, axis=0).astype(np.float32)
    y_bear = np.array(y_bear_list, dtype=np.float32)

    def add_spurious(X, y, e):
        flip = (rng.rand(len(y)) < 0.01).astype(np.float32)
        y_noisy = np.abs(y - flip)

        flip_spurious = (rng.rand(len(y_noisy)) < e).astype(np.float32)
        s = np.abs(y_noisy - flip_spurious)[:, None]

        X_full = np.concatenate([X, s], axis=1)
        return X_full, y_noisy

    # Env0: Bull
    X_bull_full, y_bull_noisy = add_spurious(X_bull, y_bull, e_bull)
    # Env1: Sideways
    X_side_full, y_side_noisy = add_spurious(X_side, y_side, e_side)
    # Env2: Bear (OOD test regime with different spurious correlation)
    X_bear_full, y_bear_noisy = add_spurious(X_bear, y_bear, e_bear)

    env0 = (X_bull_full, y_bull_noisy)
    env1 = (X_side_full, y_side_noisy)
    env2 = (X_bear_full, y_bear_noisy)

    return [env0, env1, env2]

###############################################################################
# Helper to build env list for training functions
###############################################################################

def build_finance_envs(prices, window=10, env_mode="time_spurious", seed=0):
    if env_mode == "time_spurious":
        T = len(prices)
        prices_env1 = prices[: T // 3]
        prices_env2 = prices[T // 3: 2 * T // 3]
        prices_env3 = prices[2 * T // 3:]

        envs = [
            make_finance_environment(prices_env1, window=window, e=0.2, seed=seed + 0),
            make_finance_environment(prices_env2, window=window, e=0.1, seed=seed + 1),
            make_finance_environment(prices_env3, window=window, e=0.9, seed=seed + 2),
        ]
        return envs

    elif env_mode == "spurious_strength":
        envs = make_finance_envs_spurious_strength(
            prices,
            window=window,
            e_list=(0.1, 0.2, 0.9),
            seed=seed,
        )
        return envs

    elif env_mode == "factor":
        envs = make_factor_envs(
            prices,
            window=window,
            seed=seed,
        )
        return envs

    elif env_mode == "trend_regimes":
        envs = make_trend_regime_envs(
            prices,
            window=window,
            short_window=50,
            long_window=200,
            side_eps=0.5,
            e_bull=0.2,
            e_side=0.2,
            e_bear=0.9,
            seed=seed,
        )
        return envs

    else:
        raise ValueError(f"Unknown env_mode: {env_mode}")


def train_erm(hidden_dim=256, l2_regularizer_weight=0.001, lr=0.001,
              n_restarts=10, steps=501, env_mode="time_spurious",
              ticker="aapl"):
    final_train_accs = []
    final_test_accs = []

    for restart in range(n_restarts):
        print(f"Restart {restart}")

        prices = download_prices(ticker, start="2010-01-01", end="2024-01-01")

        # Build envs according to env_mode
        envs = build_finance_envs(
            prices,
            window=10,
            env_mode=env_mode,
            seed=restart,
        )

        # Train on first two envs, test on third
        X_train = np.vstack([envs[0][0], envs[1][0]])
        y_train = np.hstack([envs[0][1], envs[1][1]])
        X_test, y_test = envs[2]

        d = X_train.shape[1]
        h = hidden_dim

        limit_W = np.sqrt(6.0 / (d + h))
        W = np.random.uniform(-limit_W, limit_W, (h, d)).astype(np.float32)

        limit_V = np.sqrt(6.0 / (h + 1))
        V = np.random.uniform(-limit_V, limit_V, (1, h)).astype(np.float32)

        m_W = np.zeros_like(W)
        m_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        v_V = np.zeros_like(V)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        print("step         train nll    train acc    test acc")

        for step in range(steps):
            Z_train, H_train, F_train = forward(X_train, W, V)
            train_nll = mean_nll(F_train[:, 0], y_train)
            train_acc = mean_accuracy(F_train[:, 0], y_train)

            Z_test, H_test, F_test = forward(X_test, W, V)
            test_acc = mean_accuracy(F_test[:, 0], y_test)

            dW_nll, dV_nll = gradients_nll(X_train, y_train,
                                           Z_train, H_train, F_train, W, V)

            total_grad_W = dW_nll + l2_regularizer_weight * 2 * W
            total_grad_V = dV_nll + l2_regularizer_weight * 2 * V

            m_W = beta1 * m_W + (1 - beta1) * total_grad_W
            m_V = beta1 * m_V + (1 - beta1) * total_grad_V
            v_W = beta2 * v_W + (1 - beta2) * (total_grad_W ** 2)
            v_V = beta2 * v_V + (1 - beta2) * (total_grad_V ** 2)

            m_W_hat = m_W / (1 - beta1 ** (step + 1))
            m_V_hat = m_V / (1 - beta1 ** (step + 1))
            v_W_hat = v_W / (1 - beta2 ** (step + 1))
            v_V_hat = v_V / (1 - beta2 ** (step + 1))

            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

            if step % 100 == 0:
                print(f"{step:<12} {train_nll:<12.5f} {train_acc:<12.5f} {test_acc:<12.5f}")

        final_train_accs.append(train_acc)
        final_test_accs.append(test_acc)

        print(f"Final train acc (mean/std across restarts so far):")
        print(f"{np.mean(final_train_accs):.5f} {np.std(final_train_accs):.5f}")
        print(f"Final test acc (mean/std across restarts so far):")
        print(f"{np.mean(final_test_accs):.5f} {np.std(final_test_accs):.5f}")
        print()

    return final_train_accs, final_test_accs


def train_irm(hidden_dim=256, l2_regularizer_weight=0.001, lr=0.001,
              n_restarts=10, penalty_anneal_iters=100, penalty_weight=10000.0,
              steps=501, env_mode="time_spurious", ticker="aapl"):
    final_train_accs = []
    final_test_accs = []

    for restart in range(n_restarts):
        print(f"Restart {restart}")

        prices = download_prices(ticker, start="2010-01-01", end="2024-01-01")

        envs = build_finance_envs(
            prices,
            window=10,
            env_mode=env_mode,
            seed=restart,
        )

        X_env0, y_env0 = envs[0]
        d = X_env0.shape[1]
        h = hidden_dim

        limit_W = np.sqrt(6.0 / (d + h))
        W = np.random.uniform(-limit_W, limit_W, (h, d)).astype(np.float32)

        limit_V = np.sqrt(6.0 / (h + 1))
        V = np.random.uniform(-limit_V, limit_V, (1, h)).astype(np.float32)

        m_W = np.zeros_like(W)
        m_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        v_V = np.zeros_like(V)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        print("step         train nll    train acc    train penalty test acc")

        for step in range(steps):
            env_nlls = []
            env_accs = []
            env_penalties = []

            for env_idx in range(len(envs)):
                X_env, y_env = envs[env_idx]
                Z, H, F = forward(X_env, W, V)

                nll = mean_nll(F[:, 0], y_env)
                acc = mean_accuracy(F[:, 0], y_env)
                pen = penalty(F[:, 0], y_env)

                env_nlls.append(nll)
                env_accs.append(acc)
                env_penalties.append(pen)

            train_nll = np.mean(env_nlls[:2])
            train_acc = np.mean(env_accs[:2])
            train_penalty = np.mean(env_penalties[:2])
            # here we keep the 3rd env as "test" env
            test_acc = env_accs[2]

            pw = penalty_weight if step >= penalty_anneal_iters else 1.0

            total_grad_W = np.zeros_like(W)
            total_grad_V = np.zeros_like(V)

            for env_idx in [0, 1]:
                X_env, y_env = envs[env_idx]
                Z, H, F = forward(X_env, W, V)

                dW_nll, dV_nll = gradients_nll(X_env, y_env, Z, H, F, W, V)
                dW_penalty, dV_penalty = compute_penalty_gradients(
                    X_env, y_env, Z, H, F, W, V
                )

                total_grad_W += dW_nll + pw * dW_penalty
                total_grad_V += dV_nll + pw * dV_penalty

            total_grad_W /= 2.0
            total_grad_V /= 2.0

            total_grad_W += l2_regularizer_weight * 2 * W
            total_grad_V += l2_regularizer_weight * 2 * V

            if pw > 1.0:
                total_grad_W /= pw
                total_grad_V /= pw

            m_W = beta1 * m_W + (1 - beta1) * total_grad_W
            m_V = beta1 * m_V + (1 - beta1) * total_grad_V
            v_W = beta2 * v_W + (1 - beta2) * (total_grad_W ** 2)
            v_V = beta2 * v_V + (1 - beta2) * (total_grad_V ** 2)

            m_W_hat = m_W / (1 - beta1 ** (step + 1))
            m_V_hat = m_V / (1 - beta1 ** (step + 1))
            v_W_hat = v_W / (1 - beta2 ** (step + 1))
            v_V_hat = v_V / (1 - beta2 ** (step + 1))

            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

            if step % 100 == 0:
                print(f"{step:<12} {train_nll:<12.5f} {train_acc:<12.5f} "
                      f"{train_penalty:<13.5f} {test_acc:<12.5f}")

        final_train_accs.append(train_acc)
        final_test_accs.append(test_acc)

        print(f"Final train acc (mean/std across restarts so far):")
        print(f"{np.mean(final_train_accs):.5f} {np.std(final_train_accs):.5f}")
        print(f"Final test acc (mean/std across restarts so far):")
        print(f"{np.mean(final_test_accs):.5f} {np.std(final_test_accs):.5f}")
        print()

    return final_train_accs, final_test_accs


def train_irm_game(hidden_dim=256, 
                   l2_regularizer_weight=0.001, 
                   lr=2.5e-4,
                   n_restarts=10, 
                   steps=500,
                   batch_size=256,
                   warm_start=True,
                   env_mode="time_spurious",
                   ticker="aapl"):
    
    final_train_accs = []
    final_test_accs = []
    
    beta1 = 0.9
    beta2 = 0.999
    eps   = 1e-8

    for restart in range(n_restarts):
        print(f"Restart {restart}")

        prices = download_prices(ticker, start="2010-01-01", end="2024-01-01")

        envs = build_finance_envs(
            prices,
            window=10,
            env_mode=env_mode,
            seed=restart,
        )

        num_train_envs = 2

        X_env0, y_env0 = envs[0]
        d = X_env0.shape[1]
        h = hidden_dim

        limit_W = np.sqrt(6.0 / (d + h))
        W = np.random.uniform(-limit_W, limit_W,
                              (num_train_envs, h, d)).astype(np.float32)
        limit_V = np.sqrt(6.0 / (h + 1))
        V = np.random.uniform(-limit_V, limit_V,
                              (num_train_envs, 1, h)).astype(np.float32)

        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        t_env = np.zeros(num_train_envs, dtype=np.int64)
        train_envs = envs[:num_train_envs]

        # Warm Start
        if warm_start:
            total_train_points = sum(X.shape[0] for (X, _) in train_envs)
            steps_warm = max(1, total_train_points // batch_size)

            for ws in range(steps_warm):
                for env_idx in range(num_train_envs):
                    X_e, y_e = train_envs[env_idx]
                    N_e = X_e.shape[0]
                    bsz = min(batch_size, N_e)
                    idx = np.random.randint(0, N_e, size=bsz)
                    Xb = X_e[idx]
                    yb = y_e[idx]

                    Z, H, F = forward(Xb, W[env_idx], V[env_idx])
                    dW, dV = gradients_nll(Xb, yb, Z, H, F, W[env_idx], V[env_idx])
                    dW += 2 * l2_regularizer_weight * W[env_idx]
                    dV += 2 * l2_regularizer_weight * V[env_idx]
                    t_env[env_idx] += 1
                    t = t_env[env_idx]

                    m_W[env_idx] = beta1 * m_W[env_idx] + (1 - beta1) * dW
                    v_W[env_idx] = beta2 * v_W[env_idx] + (1 - beta2) * (dW ** 2)
                    m_V[env_idx] = beta1 * m_V[env_idx] + (1 - beta1) * dV
                    v_V[env_idx] = beta2 * v_V[env_idx] + (1 - beta2) * (dV ** 2)

                    m_W_hat = m_W[env_idx] / (1 - beta1 ** t)
                    v_W_hat = v_W[env_idx] / (1 - beta2 ** t)
                    m_V_hat = m_V[env_idx] / (1 - beta1 ** t)
                    v_V_hat = v_V[env_idx] / (1 - beta2 ** t)

                    W[env_idx] -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                    V[env_idx] -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

        print("step         train nll    train acc    train penalty test acc")

        for step in range(steps):

            for env_idx in range(num_train_envs):
                X_e, y_e = train_envs[env_idx]
                N_e = X_e.shape[0]
                bsz = min(batch_size, N_e)
                idx = np.random.randint(0, N_e, size=bsz)
                Xb = X_e[idx]
                yb = y_e[idx]

                B = Xb.shape[0]
                logits_sum = np.zeros(B, dtype=np.float32)
                Z_env = None
                H_env = None

                for e in range(num_train_envs):
                    if e == env_idx:
                        Z_env, H_env, F_env = forward(Xb, W[e], V[e])
                        logits_sum += F_env[:, 0]
                    else:
                        _, _, F_other = forward(Xb, W[e], V[e])
                        logits_sum += F_other[:, 0]

                ensemble_logits_env = logits_sum / float(num_train_envs)

                dW_env, dV_env = gradients_nll_ensemble_env(
                    Xb, yb,
                    Z_env, H_env,
                    ensemble_logits_env,
                    W[env_idx], V[env_idx],
                    num_envs=num_train_envs
                )
                dW_env += 2 * l2_regularizer_weight * W[env_idx]
                dV_env += 2 * l2_regularizer_weight * V[env_idx]
                t_env[env_idx] += 1
                t = t_env[env_idx]

                m_W[env_idx] = beta1 * m_W[env_idx] + (1 - beta1) * dW_env
                v_W[env_idx] = beta2 * v_W[env_idx] + (1 - beta2) * (dW_env ** 2)
                m_V[env_idx] = beta1 * m_V[env_idx] + (1 - beta1) * dV_env
                v_V[env_idx] = beta2 * v_V[env_idx] + (1 - beta2) * (dV_env ** 2)

                m_W_hat = m_W[env_idx] / (1 - beta1 ** t)
                v_W_hat = v_W[env_idx] / (1 - beta2 ** t)
                m_V_hat = m_V[env_idx] / (1 - beta1 ** t)
                v_V_hat = v_V[env_idx] / (1 - beta2 ** t)

                W[env_idx] -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                V[env_idx] -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

            env_nlls = []
            env_accs = []
            env_penalties = []

            for e_idx in range(len(envs)):
                X_env, y_env = envs[e_idx]
                logits_env = ensemble_logits_on_env(X_env, W, V)
                nll = mean_nll(logits_env, y_env)
                acc = mean_accuracy(logits_env, y_env)
                pen = penalty(logits_env, y_env)
                env_nlls.append(nll)
                env_accs.append(acc)
                env_penalties.append(pen)

            train_nll = np.mean(env_nlls[:num_train_envs])
            train_acc = np.mean(env_accs[:num_train_envs])
            train_penalty = np.mean(env_penalties[:num_train_envs])
            test_acc = env_accs[num_train_envs]

            if step % 50 == 0:
                print(f"{step:<12} {train_nll:<12.5f} {train_acc:<12.5f} "
                      f"{train_penalty:<13.5f} {test_acc:<12.5f}")

        final_train_accs.append(train_acc)
        final_test_accs.append(test_acc)

        print("Final train acc (mean/std across restarts so far):")
        print(f"{np.mean(final_train_accs):.5f} {np.std(final_train_accs):.5f}")
        print("Final test acc (mean/std across restarts so far):")
        print(f"{np.mean(final_test_accs):.5f} {np.std(final_test_accs):.5f}")
        print()

    return final_train_accs, final_test_accs


if __name__ == "__main__":
    # Options: "time_spurious", "spurious_strength", "factor", "trend_regimes"
    ENV_MODE = "trend_regimes"
    # For trend regimes, SPY is a nice market index; you can change to "aapl" etc.
    TICKER = "SPY"

    print(f"Finance (yfinance): ERM vs IRM vs IRM Games [{ENV_MODE}]")

    print("\n Training ERM")
    erm_train_accs, erm_test_accs = train_erm(
        hidden_dim=256,
        l2_regularizer_weight=0.001,
        lr=0.001,
        n_restarts=10,
        steps=501,
        env_mode=ENV_MODE,
        ticker=TICKER,
    )

    print("\n Training IRM")
    irm_train_accs, irm_test_accs = train_irm(
        hidden_dim=256,
        l2_regularizer_weight=0.001,
        lr=0.001,
        n_restarts=10,
        penalty_anneal_iters=100,
        penalty_weight=10000.0,
        steps=501,
        env_mode=ENV_MODE,
        ticker=TICKER,
    )

    print("\n Training IRM Games")
    irm_game_train_accs, irm_game_test_accs = train_irm_game(
        hidden_dim=256, 
        l2_regularizer_weight=0.001, 
        lr=2.5e-4,
        n_restarts=10, 
        steps=501,
        batch_size=256,
        warm_start=True,
        env_mode=ENV_MODE,
        ticker=TICKER,
    )

    print(f"\nERM Results:")
    print(f"  Train: {np.mean(erm_train_accs):.1%} ± {np.std(erm_train_accs):.1%}")
    print(f"  Test:  {np.mean(erm_test_accs):.1%} ± {np.std(erm_test_accs):.1%}")
    
    print(f"\nIRM Results:")
    print(f"  Train: {np.mean(irm_train_accs):.1%} ± {np.std(irm_train_accs):.1%}")
    print(f"  Test:  {np.mean(irm_test_accs):.1%} ± {np.std(irm_test_accs):.1%}")
    
    print(f"\nIRM Games Results:")
    print(f"  Train: {np.mean(irm_game_train_accs):.1%} ± {np.std(irm_game_train_accs):.1%}")
    print(f"  Test:  {np.mean(irm_game_test_accs):.1%} ± {np.std(irm_game_test_accs):.1%}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    methods = ['ERM', 'IRM', 'IRM Games']
    train_means = [
        np.mean(erm_train_accs),
        np.mean(irm_train_accs),
        np.mean(irm_game_train_accs),
    ]
    test_means = [
        np.mean(erm_test_accs),
        np.mean(irm_test_accs),
        np.mean(irm_game_test_accs),
    ]
    train_stds = [
        np.std(erm_train_accs),
        np.std(irm_train_accs),
        np.std(irm_game_train_accs),
    ]
    test_stds = [
        np.std(erm_test_accs),
        np.std(irm_test_accs),
        np.std(irm_game_test_accs),
    ]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax1.bar(x - width/2, train_means, width, yerr=train_stds, label='Train', capsize=5)
    ax1.bar(x + width/2, test_means, width, yerr=test_stds, label='Test', capsize=5)
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'ERM vs IRM vs IRM Games ({ENV_MODE}, {TICKER})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    erm_gap = np.mean(erm_train_accs) - np.mean(erm_test_accs)
    irm_gap = np.mean(irm_train_accs) - np.mean(irm_test_accs)
    irm_game_gap = np.mean(irm_game_train_accs) - np.mean(irm_game_test_accs)
    
    ax2.bar(methods, [erm_gap, irm_gap, irm_game_gap])
    ax2.set_ylabel('Train - Test Accuracy')
    ax2.set_title('Generalization Gap')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('erm_irm_irmgames_finance_trend_regimes.png', dpi=150, bbox_inches='tight')
