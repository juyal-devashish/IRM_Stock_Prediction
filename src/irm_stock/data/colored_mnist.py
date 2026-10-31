"""ColoredMNIST (Arjovsky et al., 2019).

Digits are binarised (< 5 vs >= 5) with 25% label noise and coloured red/green so
colour agrees with the label with probability ``1 - e``. Colour is therefore more
predictive than shape in the training environments (e = 0.2, 0.1) but reversed in
the test environment (e = 0.9).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from irm_stock.types import Environment

N_TRAIN = 50_000
TRAIN_ENV_E = (0.2, 0.1)
TEST_ENV_E = 0.9


def default_data_dir() -> Path:
    return Path(os.environ.get("IRM_STOCK_DATA_DIR", Path.home() / "datasets" / "mnist"))


def load_mnist(data_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the 60k MNIST training images (uint8, (n, 28, 28)) and labels.

    Requires the ``mnist`` extra (``uv sync --extra mnist``) for torchvision.
    """
    try:
        from torchvision import datasets
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("ColoredMNIST needs torchvision: run `uv sync --extra mnist`.") from exc
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    mnist = datasets.MNIST(str(root.expanduser()), train=True, download=True)
    return mnist.data.numpy(), mnist.targets.numpy()


def make_environment(
    images: np.ndarray, labels: np.ndarray, e: float, rng: np.random.RandomState
) -> Environment:
    """Downsample to 14x14, binarise + noise the label, colour by (noisy) label."""
    images = images.reshape(-1, 28, 28)[:, ::2, ::2]
    labels = (labels < 5).astype(np.float32)
    labels = np.abs(labels - (rng.rand(len(labels)) < 0.25).astype(np.float32))
    colors = np.abs(labels - (rng.rand(len(labels)) < e).astype(np.float32))

    images_colored = np.stack([images, images], axis=1).astype(np.float32)
    images_colored[np.arange(len(images)), 1 - colors.astype(np.int64), :, :] = 0

    X = images_colored.reshape(len(images), -1).astype(np.float32) / 255.0
    return X, labels


def make_colored_mnist_envs(
    images: np.ndarray, labels: np.ndarray, rng: np.random.RandomState
) -> list[Environment]:
    """Shuffle the first 50k digits, split them into two training environments, and
    use the remaining digits as the test environment. Inputs are not modified."""
    train_x, train_y = images[:N_TRAIN].copy(), labels[:N_TRAIN].copy()
    val_x, val_y = images[N_TRAIN:], labels[N_TRAIN:]

    # Shuffle images and labels with the same permutation.
    state = rng.get_state()
    rng.shuffle(train_x)
    rng.set_state(state)
    rng.shuffle(train_y)

    return [
        make_environment(train_x[::2], train_y[::2], TRAIN_ENV_E[0], rng),
        make_environment(train_x[1::2], train_y[1::2], TRAIN_ENV_E[1], rng),
        make_environment(val_x, val_y, TEST_ENV_E, rng),
    ]
