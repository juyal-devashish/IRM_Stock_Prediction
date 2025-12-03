import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(np.float32)

def softmax(F):
    F_stable = F - np.max(F, axis=1, keepdims=True)
    expF = np.exp(F_stable)
    return expF / np.sum(expF, axis=1, keepdims=True)


def forward(X, W, V):
    Z = X @ W.T
    H = relu(Z)
    F = H @ V.T
    P = softmax(F)
    return Z, H, F, P

def gradients_softmax(X, y, Z, H, F, P, W, V):
    B = X.shape[0]
    K = V.shape[0]

    Y = np.zeros((B, K), dtype=np.float32)
    Y[np.arange(B), y] = 1.0

    dF = (P - Y)
    dV = (dF.T @ H) / B
    dH = dF @ V
    dZ = dH * relu_grad(Z)
    dW = (dZ.T @ X) / B

    return dW, dV

def train_softmax(X_train, y_train, X_test, y_test,
                  h, lr=1e-3, steps=30000, batch=32):

    d = X_train.shape[1]
    K = 10

    W = np.random.randn(h, d) * np.sqrt(2.0 / d)
    V = np.random.randn(K, h) * np.sqrt(2.0 / h)

    iters = []
    train_acc_hist = []
    test_acc_hist = []

    N = len(X_train)

    for t in range(steps):
        # minibatch
        idx = np.random.randint(0, N, batch)
        Xb = X_train[idx]
        yb = y_train[idx]

        Z, H, F, P = forward(Xb, W, V)
        dW, dV = gradients_softmax(Xb, yb, Z, H, F, P, W, V)

        # SGD being updated
        W -= lr * dW
        V -= lr * dV

        if t % 100 == 0:
            # training accuracy
            _, _, _, P_train = forward(X_train, W, V)
            pred_train = np.argmax(P_train, axis=1)
            acc_train = np.mean(pred_train == y_train)

            # testing accuracy
            _, _, _, P_test = forward(X_test, W, V)
            pred_test = np.argmax(P_test, axis=1)
            acc_test = np.mean(pred_test == y_test)

            iters.append(t)
            train_acc_hist.append(acc_train)
            test_acc_hist.append(acc_test)

            print(f"h={h}, t={t:5d}: train_acc={acc_train:.4f}, test_acc={acc_test:.4f}")

    return W, V, iters, train_acc_hist, test_acc_hist


if __name__ == "__main__":
    # Load normalized MNIST
    data = np.load("mnist_normalized_bias.npz")
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"].astype(np.int64)
    y_test  = data["y_test"].astype(np.int64)

    print("Loaded MNIST for Task B.")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    results = {}

    for h in [5, 40, 200]:
        print(f"\n=== Training Softmax CE, h={h} ===")
        W, V, iters, train_hist, test_hist = train_softmax(
            X_train, y_train, X_test, y_test,
            h=h, lr=1e-3, steps=30000, batch=32
        )
        results[h] = (iters, train_hist, test_hist)
        print(f"Final test accuracy (h={h}): {test_hist[-1]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, h in zip(axes, [5, 40, 200]):
        iters, train_hist, test_hist = results[h]
        ax.plot(iters, train_hist, label="train")
        ax.plot(iters, test_hist, label="test")
        ax.set_title(f"h = {h}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("accuracy")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Softmax Cross-Entropy: Training/Test Accuracy vs Iterations")
    plt.tight_layout()
    plt.show()
