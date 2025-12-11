from util_file import *

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

### yfinance content below 

def download_prices(ticker, start="2010-01-01", end="2024-01-01"):
    data = yf.download(ticker, start=start, end=end, group_by="column")

    if data is None or len(data) == 0:
        raise ValueError(f"No data returned for {ticker}")

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

    flip = (rng.rand(len(y)) < 0.01).astype(np.float32)
    y_noisy = np.abs(y - flip)

    flip_spurious = (rng.rand(len(y_noisy)) < e).astype(np.float32)
    s = np.abs(y_noisy - flip_spurious)

    s = s[:, None]
    X_full = np.concatenate([X, s], axis=1)

    return X_full, y_noisy

def train_erm(hidden_dim=256, l2_regularizer_weight=0.001, lr=0.001,
              n_restarts=10, steps=501):
    final_train_accs = []
    final_test_accs = []

    for restart in range(n_restarts):
        print(f"Restart {restart}")

        prices = download_prices("aapl", start="2010-01-01", end="2024-01-01")
        T = len(prices)

        prices_env1 = prices[: T // 3]
        prices_env2 = prices[T // 3: 2 * T // 3]
        prices_env3 = prices[2 * T // 3:]

        envs = [
            make_finance_environment(prices_env1, window=10, e=0.2, seed=0),
            make_finance_environment(prices_env2, window=10, e=0.1, seed=1),
            make_finance_environment(prices_env3, window=10, e=0.9, seed=2)
        ]

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

            dW_nll, dV_nll = gradients_nll(X_train, y_train, Z_train, H_train, F_train, W, V)

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
              steps=501):
    final_train_accs = []
    final_test_accs = []

    for restart in range(n_restarts):
        print(f"Restart {restart}")

        prices = download_prices("aapl", start="2010-01-01", end="2024-01-01")
        T = len(prices)

        prices_env1 = prices[: T // 3]
        prices_env2 = prices[T // 3: 2 * T // 3]
        prices_env3 = prices[2 * T // 3:]

        envs = [
            make_finance_environment(prices_env1, window=10, e=0.2, seed=0),
            make_finance_environment(prices_env2, window=10, e=0.1, seed=1),
            make_finance_environment(prices_env3, window=10, e=0.9, seed=2)
        ]

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

            for env_idx in range(3):
                X_env, y_env = envs[env_idx]
                Z, H, F = forward(X_env, W, V)

                nll = mean_nll(F[:, 0], y_env)
                acc = mean_accuracy(F[:, 0], y_env)
                pen = penalty(F[:, 0], y_env)

                env_nlls.append(nll)
                env_accs.append(acc)
                env_penalties.append(pen)

            train_nll = np.mean([env_nlls[0], env_nlls[1]])
            train_acc = np.mean([env_accs[0], env_accs[1]])
            train_penalty = np.mean([env_penalties[0], env_penalties[1]])
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
                   lr=0.001,
                   n_restarts=10,
                   steps=501,
                   inner_steps=15):

    final_train_accs = []
    final_test_accs = []

    for restart in range(n_restarts):
        print(f"Restart {restart}")

        prices = download_prices("aapl", start="2010-01-01", end="2024-01-01")
        T = len(prices)

        prices_env1 = prices[: T // 3]
        prices_env2 = prices[T // 3: 2 * T // 3]
        prices_env3 = prices[2 * T // 3:]

        envs = [
            make_finance_environment(prices_env1, window=10, e=0.2, seed=0),
            make_finance_environment(prices_env2, window=10, e=0.1, seed=1),
            make_finance_environment(prices_env3, window=10, e=0.9, seed=2)
        ]

        X_env0, y_env0 = envs[0]
        d = X_env0.shape[1]
        h = hidden_dim
        num_train_envs = 2  # env 0 and 1 are players; env 2 is test

        # One classifier per *training* environment
        limit_W = np.sqrt(6.0 / (d + h))
        W = np.random.uniform(-limit_W, limit_W,
                              (num_train_envs, h, d)).astype(np.float32)

        limit_V = np.sqrt(6.0 / (h + 1))
        V = np.random.uniform(-limit_V, limit_V,
                              (num_train_envs, 1, h)).astype(np.float32)

        # Adam state (shared arrays, as you already had)
        m_W = np.zeros_like(W)
        m_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        v_V = np.zeros_like(V)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        # NEW: global Adam step counter for correct bias correction
        t = 0

        print("step         train nll    train acc    train penalty test acc")

        for step in range(steps):
            # ---------- 1) Compute metrics (same idea as before) ----------
            env_nlls = []
            env_accs = []
            env_penalties = []

            # Use the *ensemble* wav on each env (including test env)
            for env_idx in range(len(envs)):
                X_env, y_env = envs[env_idx]
                logits_env = ensemble_logits_on_env(X_env, W, V)
                nll = mean_nll(logits_env, y_env)
                acc = mean_accuracy(logits_env, y_env)
                pen = penalty(logits_env, y_env)  # just monitored, not optimized
                env_nlls.append(nll)
                env_accs.append(acc)
                env_penalties.append(pen)

            train_nll = np.mean(env_nlls[:num_train_envs])
            train_acc = np.mean(env_accs[:num_train_envs])
            train_penalty = np.mean(env_penalties[:num_train_envs])
            test_acc = np.mean(env_accs[num_train_envs:])

            # ---------- 2) Best-response updates with inner steps ----------
            # For each training environment, treat others as fixed
            for env_idx in range(num_train_envs):
                X_env, y_env = envs[env_idx]

                # Take multiple Adam steps for this player = approximate best response
                for inner in range(inner_steps):
                    B = X_env.shape[0]
                    logits_sum = np.zeros(B, dtype=np.float32)

                    Z_env = None
                    H_env = None

                    # Build ensemble logits for this env given current W, V
                    for e in range(num_train_envs):
                        if e == env_idx:
                            Z_env, H_env, F_env = forward(X_env, W[e], V[e])
                            logits_sum += F_env[:, 0]
                        else:
                            _, _, F_other = forward(X_env, W[e], V[e])
                            logits_sum += F_other[:, 0]

                    ensemble_logits_env = logits_sum / float(num_train_envs)

                    # Gradient of env_idx's risk wrt its own classifier,
                    # given the ensemble wav over all training envs.
                    dW_env, dV_env = gradients_nll_ensemble_env(
                        X_env, y_env,
                        Z_env, H_env,
                        ensemble_logits_env,
                        W[env_idx], V[env_idx],
                        num_train_envs
                    )

                    # L2 regularization on that env's parameters
                    dW_env += l2_regularizer_weight * 2 * W[env_idx]
                    dV_env += l2_regularizer_weight * 2 * V[env_idx]

                    # Pack into full gradient tensors so we can reuse your Adam code
                    total_grad_W = np.zeros_like(W)
                    total_grad_V = np.zeros_like(V)
                    total_grad_W[env_idx] = dW_env
                    total_grad_V[env_idx] = dV_env

                    # Adam update (same as before, but with global counter t)
                    t += 1
                    m_W = beta1 * m_W + (1 - beta1) * total_grad_W
                    m_V = beta1 * m_V + (1 - beta1) * total_grad_V
                    v_W = beta2 * v_W + (1 - beta2) * (total_grad_W ** 2)
                    v_V = beta2 * v_V + (1 - beta2) * (total_grad_V ** 2)

                    m_W_hat = m_W / (1 - beta1 ** t)
                    m_V_hat = m_V / (1 - beta1 ** t)
                    v_W_hat = v_W / (1 - beta2 ** t)
                    v_V_hat = v_V / (1 - beta2 ** t)

                    # Only update the active environment's parameters
                    W[env_idx] -= lr * m_W_hat[env_idx] / (np.sqrt(v_W_hat[env_idx]) + eps)
                    V[env_idx] -= lr * m_V_hat[env_idx] / (np.sqrt(v_V_hat[env_idx]) + eps)

            if step % 100 == 0:
                print(f"{step:<12} {train_nll:<12.5f} {train_acc:<12.5f} "
                      f"{train_penalty:<13.5f} {test_acc:<12.5f}")

        final_train_accs.append(train_acc)
        final_test_accs.append(test_acc)

        print(f"Final train acc (mean/std across restarts so far):")
        print(f"{np.mean(final_train_accs):.5f} {np.std(final_train_accs):.5f}")
        print(f"Final test acc (mean/std across restarts so far):")
        print(f"Final test acc (mean/std across restarts so far):")
        print(f"{np.mean(final_test_accs):.5f} {np.std(final_test_accs):.5f}")
        print()

    return final_train_accs, final_test_accs

def train_majority(n_restarts=10):
    final_train_accs = []
    final_test_accs = []

    for restart in range(n_restarts):
        print(f"Restart {restart}")

        prices = download_prices("aapl", start="2010-01-01", end="2024-01-01")
        T = len(prices)

        prices_env1 = prices[: T // 3]
        prices_env2 = prices[T // 3: 2 * T // 3]
        prices_env3 = prices[2 * T // 3:]

        envs = [
            make_finance_environment(prices_env1, window=10, e=0.2, seed=0),
            make_finance_environment(prices_env2, window=10, e=0.1, seed=1),
            make_finance_environment(prices_env3, window=10, e=0.9, seed=2)
        ]

        X_train = np.vstack([envs[0][0], envs[1][0]])
        y_train = np.hstack([envs[0][1], envs[1][1]])
        X_test, y_test = envs[2]

        maj_label = 1.0 if np.mean(y_train) >= 0.5 else 0.0

        y_train_pred = np.full_like(y_train, maj_label)
        y_test_pred = np.full_like(y_test, maj_label)

        train_acc = np.mean((y_train_pred == y_train).astype(np.float32))
        test_acc = np.mean((y_test_pred == y_test).astype(np.float32))

        final_train_accs.append(train_acc)
        final_test_accs.append(test_acc)

        print(f"Train acc: {train_acc:.5f}  Test acc: {test_acc:.5f}")
        print(f"Final train acc (mean/std across restarts so far):")
        print(f"{np.mean(final_train_accs):.5f} {np.std(final_train_accs):.5f}")
        print(f"Final test acc (mean/std across restarts so far):")
        print(f"{np.mean(final_test_accs):.5f} {np.std(final_test_accs):.5f}")
        print()

    return final_train_accs, final_test_accs

if __name__ == "__main__":
    print("Finance (yfinance): ERM vs IRM")
    
    final_train_accs, final_test_accs = train_majority(10)
    print(f"  Train: {np.mean(final_train_accs):.1%} ± {np.std(final_train_accs):.1%}")
    print(f"  Test:  {np.mean(final_test_accs):.1%} ± {np.std(final_test_accs):.1%}")
    
    print("\n Training ERM")
    erm_train_accs, erm_test_accs = train_erm(
        hidden_dim=256,
        l2_regularizer_weight=0.001,
        lr=0.001,
        n_restarts=10,
        steps=501
    )

    print("\n Training IRM")

    irm_train_accs, irm_test_accs = train_irm(
        hidden_dim=256,
        l2_regularizer_weight=0.001,
        lr=0.001,
        n_restarts=10,
        penalty_anneal_iters=100,
        penalty_weight=10000.0,
        steps=501
    )
    
    irm_game_train_accs, irm_game_test_accs = train_irm_game(
        hidden_dim=256,
        l2_regularizer_weight=0.001,
        lr=0.001,
        n_restarts=10,
        steps=501
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
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # methods = ['ERM', 'IRM']
    # train_means = [np.mean(erm_train_accs), np.mean(irm_train_accs)]
    # test_means = [np.mean(erm_test_accs), np.mean(irm_test_accs)]
    # train_stds = [np.std(erm_train_accs), np.std(irm_train_accs)]
    # test_stds = [np.std(erm_test_accs), np.std(irm_test_accs)]
    
    # x = np.arange(len(methods))
    # width = 0.35
    
    # ax1.bar(x - width/2, train_means, width, yerr=train_stds, label='Train', capsize=5)
    # ax1.bar(x + width/2, test_means, width, yerr=test_stds, label='Test', capsize=5)
    # ax1.set_ylabel('Accuracy')
    # ax1.set_title('ERM vs IRM: Train and Test Accuracy')
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(methods)
    # ax1.legend()
    # ax1.set_ylim([0, 1])
    # ax1.grid(axis='y', alpha=0.3)
    
    # erm_gap = np.mean(erm_train_accs) - np.mean(erm_test_accs)
    # irm_gap = np.mean(irm_train_accs) - np.mean(irm_test_accs)
    
    # ax2.bar(methods, [erm_gap, irm_gap], color=['red', 'green'])
    # ax2.set_ylabel('Train - Test Accuracy')
    # ax2.set_title('Generalization Gap')
    # ax2.grid(axis='y', alpha=0.3)
    # ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # plt.tight_layout()
    # plt.savefig('erm_vs_irm_comparison.png', dpi=150, bbox_inches='tight')
























# ### Stock price history
# ticker = yf.Ticker("aapl")
# data = ticker.history(period="1y")
# print(data)

# ### Downloading multiple tickers
# df = yf.download(["aapl", "MSFT", "GOOG"], period="6mo")
# print(df)

# ### Get info
# ticker = yf.Ticker("TSLA")
# print(ticker.info)            # Company info
# print(ticker.financials)      # Income statements
# print(ticker.balance_sheet)   # Balance sheet
# print(ticker.cashflow)        # Cash flow


# # 1. Frame the finance data in terms of how we want the environments setup

# # First will have to look how they do it from the paper


# # Need to do 80/10/10 data split