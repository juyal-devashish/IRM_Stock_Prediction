# hw4_q1_opt1.py
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from torchvision import datasets, transforms


def load_mnist():
    transform = transforms.ToTensor()

    mnist_train = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    mnist_test = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    X_train = mnist_train.data.numpy().astype(np.float32).reshape(-1, 784)
    y_train = mnist_train.targets.numpy().astype(np.int64)

    X_test  = mnist_test.data.numpy().astype(np.float32).reshape(-1, 784)
    y_test  = mnist_test.targets.numpy().astype(np.int64)

    return X_train, y_train, X_test, y_test


def normalize_and_add_bias(X_train, X_test):
    # compute training stats
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0

    # normalizing
    X_train_norm = (X_train - mu) / sigma
    X_test_norm = (X_test - mu) / sigma

    # bias
    X_train_final = np.hstack([X_train_norm, np.ones((X_train_norm.shape[0], 1))])
    X_test_final = np.hstack([X_test_norm, np.ones((X_test_norm.shape[0], 1))])

    return X_train_final, X_test_final, mu, sigma


def main():
    X_train, y_train, X_test, y_test = load_mnist()
    X_train_final, X_test_final, mu, sigma = normalize_and_add_bias(X_train, X_test)

    print("X_train shape:", X_train_final.shape)  # shape = (60000, 785)
    print("X_test shape :", X_test_final.shape)   # shape = (10000, 785)

    np.savez_compressed(
        "mnist_normalized_bias.npz",
        X_train=X_train_final,
        X_test=X_test_final,
        y_train=y_train,
        y_test=y_test,
        mean=mu,
        std=sigma
    )

    print("Saved mnist_normalized_bias.npz")


if __name__ == "__main__":
    main()
