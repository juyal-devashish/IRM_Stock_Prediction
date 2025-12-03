import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(np.float32)


def forward(X, W, V):
    """
    X: (B, d)
    W: (h, d)
    V: (1, h)
    """
    Z = X @ W.T         # (B, h)
    H = relu(Z)         # (B, h)
    F = H @ V.T         # (B, 1)
    return Z, H, F


def gradients_quadratic(X, y, Z, H, F, W, V):
    """
    Compute gradients of quadratic loss:
        L = 1/2 (y - f)^2
    """
    B = X.shape[0]

    # g = dL/df = (f - y)
    g = (F[:, 0] - y)[:, None]  # (B, 1)

    # dV: (1, h)
    dV = (g.T @ H) / B

    # dW: (h, d)
    dH = g @ V                  # (B, h)
    dZ = dH * relu_grad(Z)      # (B, h)
    dW = (dZ.T @ X) / B

    return dW, dV


def train_quadratic(X_train, y_train, X_test, y_test,
                    h, lr=1e-3, steps=30000, batch=16):
    """
    Train the network using quadratic loss.
    Returns: learned W, V, iteration list, train acc list, test acc list
    """
    d = X_train.shape[1]

    # He initialization
    W = np.random.randn(h, d) * np.sqrt(2.0 / d)
    V = np.random.randn(1, h) * np.sqrt(2.0 / h)

    iters = []
    train_acc_hist = []
    test_acc_hist = []

    N = len(X_train)

    for t in range(steps):
        # minibatch
        idx = np.random.randint(0, N, batch)
        Xb = X_train[idx]
        yb = y_train[idx]

        # forward + gradients
        Z, H, F = forward(Xb, W, V)
        dW, dV = gradients_quadratic(Xb, yb, Z, H, F, W, V)

        # SGD update
        W -= lr * dW
        V -= lr * dV

        # record accuracy every 100 steps
        if t % 100 == 0:
            # train accuracy
            _, _, F_train = forward(X_train, W, V)
            pred_train = (F_train[:, 0] > 0.5)
            acc_train = np.mean(pred_train == y_train)

            # test accuracy
            _, _, F_test = forward(X_test, W, V)
            pred_test = (F_test[:, 0] > 0.5)
            acc_test = np.mean(pred_test == y_test)

            iters.append(t)
            train_acc_hist.append(acc_train)
            test_acc_hist.append(acc_test)

            print(f"h={h}, t={t:5d}: train_acc={acc_train:.4f}, test_acc={acc_test:.4f}")

    return W, V, iters, train_acc_hist, test_acc_hist


if __name__ == "__main__":
    # Load normalized data from Part 1
    data = np.load("mnist_normalized_bias.npz")
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train_raw = data["y_train"]
    y_test_raw  = data["y_test"]

    # Convert labels
    y_train = (y_train_raw > 4).astype(np.float32)
    y_test  = (y_test_raw  > 4).astype(np.float32)

    print("Loaded MNIST (normalized) from file.")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    results = {}

    for h in [5, 40, 200]:
        print(f"\n=== Training with quadratic loss, h={h} ===")
        W, V, iters, train_hist, test_hist = train_quadratic(
            X_train, y_train, X_test, y_test,
            h=h, lr=1e-3, steps=30000, batch=16
        )
        results[h] = (iters, train_hist, test_hist)
        print(f"Final test accuracy (h={h}): {test_hist[-1]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, h in zip(axes, [5, 40, 200]):
        iters, train_hist, test_hist = results[h]
        ax.plot(iters, train_hist, label="train acc")
        ax.plot(iters, test_hist, label="test acc")
        ax.set_title(f"h = {h}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("accuracy")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Quadratic loss: training/test accuracy vs iterations")
    plt.tight_layout()
    plt.show()
