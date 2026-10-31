import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

import warnings
warnings.filterwarnings('ignore')

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



def make_environment(images, labels, e):
    images = images.reshape(-1, 28, 28)[:, ::2, ::2]
    labels = (labels < 5).astype(np.float32)
    labels = np.abs(labels - (np.random.rand(len(labels)) < 0.25).astype(np.float32))
    colors = np.abs(labels - (np.random.rand(len(labels)) < e).astype(np.float32))

    images_colored = np.stack([images, images], axis=1).astype(np.float32)
    colors_int = colors.astype(np.int64)
    channel_to_zero = 1 - colors_int
    idx = np.arange(len(images))
    images_colored[idx, channel_to_zero, :, :] = 0

    X = images_colored.reshape(len(images), -1).astype(np.float32) / 255.0
    y = labels

    return X, y


def train_erm(hidden_dim=256, l2_regularizer_weight=0.001, lr=0.001,
              n_restarts=10, steps=501):
    
    final_train_accs = []
    final_test_accs = []

    for restart in range(n_restarts):
        print(f"Restart {restart}")

        mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
        mnist_train = (mnist.data[:50000].numpy(), mnist.targets[:50000].numpy())
        mnist_val = (mnist.data[50000:].numpy(), mnist.targets[50000:].numpy())

        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0])
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1])

        envs = [
            make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
            make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
            make_environment(mnist_val[0], mnist_val[1], 0.9)
        ]
        
        X_train = np.vstack([envs[0][0], envs[1][0]])
        y_train = np.hstack([envs[0][1], envs[1][1]])
        X_test, y_test = envs[2]

        d = 2 * 14 * 14
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

        # Load MNIST
        mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
        mnist_train = (mnist.data[:50000].numpy(), mnist.targets[:50000].numpy())
        mnist_val = (mnist.data[50000:].numpy(), mnist.targets[50000:].numpy())

        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0])
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1])

        envs = [
            make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
            make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
            make_environment(mnist_val[0], mnist_val[1], 0.9)
        ]

        d = 2 * 14 * 14
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
                   steps=500,
                   batch_size=256,
                   warm_start=False):
    
    final_train_accs = []
    final_test_accs = []
    
    beta1 = 0.9
    beta2 = 0.999
    eps   = 1e-8

    for restart in range(n_restarts):
        print(f"Restart {restart}")

        mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
        mnist_train = (mnist.data[:50000].numpy(), mnist.targets[:50000].numpy())
        mnist_val = (mnist.data[50000:].numpy(), mnist.targets[50000:].numpy())

        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0])
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1])

        envs = [
            make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
            make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
            make_environment(mnist_val[0], mnist_val[1], 0.9)
        ]

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
    RUN_TRAINING = True  # set to False to skip training and just load + plot

    if RUN_TRAINING:
        print("Coloured MNIST: ERM vs IRM vs IRM-Games")

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

        print("\n Training IRM-games")
        
        irm_games_train_accs, irm_games_test_accs = train_irm_game(
            hidden_dim=256,
            l2_regularizer_weight=0.001, 
            lr=0.001,
            n_restarts=10, 
            steps=500,
            batch_size=256,
            warm_start=False
        )
        
        np.savez(
            "results_cm.npz",
            erm_train=erm_train_accs,
            erm_test=erm_test_accs,
            irm_train=irm_train_accs,
            irm_test=irm_test_accs,
            irmg_train=irm_games_train_accs,
            irmg_test=irm_games_test_accs
        )

    else:
        print("Loading saved results from results_cm.npz")
        data = np.load("results_cm.npz")
        erm_train_accs = data["erm_train"]
        erm_test_accs = data["erm_test"]
        irm_train_accs = data["irm_train"]
        irm_test_accs = data["irm_test"]
        irm_games_train_accs = data["irmg_train"]
        irm_games_test_accs = data["irmg_test"]

    print(f"\nERM Results:")
    print(f"  Train: {np.mean(erm_train_accs):.1%} ± {np.std(erm_train_accs):.1%}")
    print(f"  Test:  {np.mean(erm_test_accs):.1%} ± {np.std(erm_test_accs):.1%}")

    print(f"\nIRM Results:")
    print(f"  Train: {np.mean(irm_train_accs):.1%} ± {np.std(irm_train_accs):.1%}")
    print(f"  Test:  {np.mean(irm_test_accs):.1%} ± {np.std(irm_test_accs):.1%}")

    print(f"\nIRM-Games Results:")
    print(f"  Train: {np.mean(irm_games_train_accs):.1%} ± {np.std(irm_games_train_accs):.1%}")
    print(f"  Test:  {np.mean(irm_games_test_accs):.1%} ± {np.std(irm_games_test_accs):.1%}")

    fig, ax = plt.subplots(figsize=(10, 5))

    methods = ['ERM', 'IRM', 'IRM-Games']

    train_means = [
        np.mean(erm_train_accs),
        np.mean(irm_train_accs),
        np.mean(irm_games_train_accs)
    ]
    test_means = [
        np.mean(erm_test_accs),
        np.mean(irm_test_accs),
        np.mean(irm_games_test_accs)
    ]

    train_stds = [
        np.std(erm_train_accs),
        np.std(irm_train_accs),
        np.std(irm_games_train_accs)
    ]
    test_stds = [
        np.std(erm_test_accs),
        np.std(irm_test_accs),
        np.std(irm_games_test_accs)
    ]

    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width/2, train_means, width, yerr=train_stds, label='Train', capsize=5)
    ax.bar(x + width/2, test_means, width, yerr=test_stds, label='Test', capsize=5)

    ax.set_ylabel('Accuracy')
    ax.set_title('ERM vs IRM vs IRM-Games: Train and Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('erm_irm_irmgames_comparison.png', dpi=150, bbox_inches='tight')