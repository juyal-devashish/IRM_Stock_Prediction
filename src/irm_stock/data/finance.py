"""Stock-price download and construction of finance "environments" for IRM.

Every environment is a binary classification task: given the previous ``window``
daily returns (plus, in most modes, a *spurious* feature), predict whether the next
day's return is positive. Environments differ in how strongly the spurious feature
tracks the label, so a model that relies on it fails on the held-out environment.

:func:`build_finance_envs` always returns three environments: the first two are
for training, the third is the out-of-distribution test environment.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from irm_stock.types import Environment

logger = logging.getLogger(__name__)

#: Supported environment constructions and a one-line description of each.
ENV_MODES: dict[str, str] = {
    "time_spurious": "Price history split into thirds over time; spurious-feature "
    "agreement 80% / 90% / 10% (test).",
    "spurious_strength": "Same samples in every environment; only the spurious-feature "
    "strength changes (90% / 80% / 10% agreement).",
    "factor": "Two features: invariant mean return, and a mix of mean and volatility "
    "whose sign flips in the test environment.",
    "trend_regimes": "Bull / sideways / bear regimes from the 50- vs 200-day moving "
    "average gap; tested on bear with a flipped spurious feature.",
    "volatility": "Low / mid / high 20-day volatility terciles; tested on high "
    "volatility with a flipped spurious feature.",
}


# --------------------------------------------------------------------------- download


def _cache_dir() -> Path:
    return Path(os.environ.get("IRM_STOCK_CACHE_DIR", Path.home() / ".cache" / "irm_stock"))


def _fetch_close_prices(ticker: str, start: str, end: str) -> np.ndarray:
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, group_by="column", progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"No price data returned for {ticker!r} ({start} to {end})")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            " ".join(str(c) for c in col if c not in ("", None)) for col in data.columns
        ]

    # Prefer adjusted close; newer yfinance (auto_adjust=True) only returns "Close".
    cols = list(data.columns)
    target = next(
        (c for cand in ("Adj Close", "Close") for c in cols if cand in c),
        cols[-1],
    )
    return data[target].to_numpy().astype(np.float32)


def download_prices(
    ticker: str,
    start: str = "2010-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
) -> np.ndarray:
    """Daily (adjusted) close prices as float32, via yfinance.

    Ranges that end in the past are cached on disk under ``$IRM_STOCK_CACHE_DIR``
    (default ``~/.cache/irm_stock``) because historical data does not change.
    """
    ticker = ticker.upper()
    cacheable = use_cache and dt.date.fromisoformat(end) <= dt.date.today()
    path = _cache_dir() / "prices" / f"{ticker}_{start}_{end}.npy"
    if cacheable and path.exists():
        logger.debug("Loading cached prices from %s", path)
        return np.load(path)

    logger.info("Downloading %s prices %s to %s", ticker, start, end)
    prices = _fetch_close_prices(ticker, start, end)
    if cacheable:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, prices)
    return prices


# ------------------------------------------------------------------- environments


def _rng(seed: int | np.random.RandomState) -> np.random.RandomState:
    return seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)


def _returns_windows_labels(
    prices: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (returns, X, y).

    ``X[i]`` holds the ``window`` returns ending at day ``t = window + i`` (exclusive)
    and ``y[i]`` is whether the return at day ``t + 1`` is positive.
    """
    prices = np.asarray(prices, dtype=np.float32)
    returns = prices[1:] / prices[:-1] - 1
    labels = (returns[1:] > 0).astype(np.float32)
    n = len(returns) - 1 - window
    if n <= 0:
        raise ValueError(f"need more than {window + 2} prices to build windows of {window}")
    X = np.lib.stride_tricks.sliding_window_view(returns, window)[:n].astype(np.float32)
    return returns, X, labels[window:]


def _add_spurious_feature(
    X: np.ndarray, y: np.ndarray, e: float, label_noise: float, rng: np.random.RandomState
) -> Environment:
    """Flip labels with prob ``label_noise``, then append a feature equal to the
    (noisy) label flipped with probability ``e``."""
    if label_noise and label_noise > 0:
        flip = (rng.rand(len(y)) < float(label_noise)).astype(np.float32)
        y = np.abs(y - flip)
    flip_spurious = (rng.rand(len(y)) < float(e)).astype(np.float32)
    s = np.abs(y - flip_spurious)[:, None]
    return np.concatenate([X, s], axis=1), y


def make_finance_environment(
    prices: np.ndarray,
    window: int = 10,
    e: float = 0.2,
    seed: int | np.random.RandomState = 0,
    label_noise: float = 0.01,
) -> Environment:
    """Return-window features plus a spurious feature that disagrees with the label
    with probability ``e``."""
    _, X, y = _returns_windows_labels(prices, window)
    return _add_spurious_feature(X, y, e, label_noise, _rng(seed))


def make_time_split_envs(
    prices: np.ndarray,
    window: int = 10,
    e_list: tuple[float, float, float] = (0.2, 0.1, 0.9),
    seed: int = 0,
    label_noise: float = 0.01,
) -> list[Environment]:
    T = len(prices)
    chunks = [prices[: T // 3], prices[T // 3 : 2 * T // 3], prices[2 * T // 3 :]]
    return [
        make_finance_environment(chunk, window, e, seed=seed + i, label_noise=label_noise)
        for i, (chunk, e) in enumerate(zip(chunks, e_list, strict=True))
    ]


def make_spurious_strength_envs(
    prices: np.ndarray,
    window: int = 10,
    e_list: tuple[float, ...] = (0.1, 0.2, 0.9),
    label_noise: float = 0.01,
    seed: int = 0,
) -> list[Environment]:
    rng = _rng(seed)
    _, X, y = _returns_windows_labels(prices, window)
    return [_add_spurious_feature(X, y.copy(), e, label_noise, rng) for e in e_list]


def make_factor_envs(prices: np.ndarray, window: int = 10) -> list[Environment]:
    """Feature 1 = mean of the return window (invariant). Feature 2 mixes the mean
    with the window volatility; its relation to the label reverses in the test env."""
    prices = np.asarray(prices, dtype=np.float32)
    returns = prices[1:] / prices[:-1] - 1
    f1_list, f2_list = [], []
    for t in range(window, len(returns)):
        win = returns[t - window : t]
        f1_list.append(win.mean())
        f2_list.append(win.std() + 1e-6)
    f1, f2 = np.array(f1_list), np.array(f2_list)
    y = (returns[window:] > 0).astype(np.float32)
    n = min(len(f1), len(y))
    f1, f2, y = f1[:n], f2[:n], y[:n]

    def env(alpha: float, beta: float) -> Environment:
        return np.stack([f1, alpha * f1 + beta * f2], axis=1).astype(np.float32), y

    return [env(1.0, 0.5), env(0.8, -0.7), env(-1.0, 0.5)]


def make_trend_regime_envs(
    prices: np.ndarray,
    window: int = 10,
    short_window: int = 50,
    long_window: int = 200,
    side_eps: float = 0.5,
    e_bull: float = 0.2,
    e_side: float = 0.2,
    e_bear: float = 0.9,
    seed: int = 0,
    label_noise: float = 0.01,
) -> list[Environment]:
    """Environments by market regime: bull, sideways (train) and bear (test).

    The regime at day ``t`` is the sign of the short-minus-long moving-average gap;
    gaps within ``side_eps`` standard deviations of zero count as sideways.
    """
    rng = _rng(seed)
    prices = np.asarray(prices, dtype=np.float32)
    series = pd.Series(prices)
    trend = (series.rolling(short_window).mean() - series.rolling(long_window).mean()).values

    returns = prices[1:] / prices[:-1] - 1
    labels = (returns[1:] > 0).astype(np.float32)

    valid = trend[~np.isnan(trend)]
    if len(valid) == 0:
        raise ValueError("Trend is all NaN; use a longer date range or shorter MA windows.")
    side_thresh = side_eps * np.std(valid)

    buckets: dict[str, tuple[list[np.ndarray], list[float]]] = {
        "bull": ([], []),
        "side": ([], []),
        "bear": ([], []),
    }
    for t in range(window, len(returns) - 1):
        tr = trend[t + 1]
        if np.isnan(tr):
            continue
        regime = "bull" if tr > side_thresh else "bear" if tr < -side_thresh else "side"
        buckets[regime][0].append(returns[t - window : t])
        buckets[regime][1].append(labels[t])

    if any(len(xs) == 0 for xs, _ in buckets.values()):
        raise ValueError(
            "Not enough samples in bull/sideways/bear regimes; "
            "try a different date range, moving-average windows, or side_eps."
        )

    envs = []
    for regime, e in (("bull", e_bull), ("side", e_side), ("bear", e_bear)):
        xs, ys = buckets[regime]
        X = np.stack(xs, axis=0).astype(np.float32)
        y = np.array(ys, dtype=np.float32)
        envs.append(_add_spurious_feature(X, y, e, label_noise, rng))
    return envs


def make_volatility_envs(
    prices: np.ndarray,
    window: int = 10,
    seed: int = 0,
    label_noise: float = 0.01,
    e_low: float = 0.2,
    e_mid: float = 0.1,
    e_high: float = 0.9,
    vol_window: int = 20,
) -> list[Environment]:
    """Environments by realised-volatility tercile: low, mid (train) and high (test)."""
    envs_full = [
        make_finance_environment(prices, window, e, seed=seed + i, label_noise=label_noise)
        for i, e in enumerate((e_low, e_mid, e_high))
    ]
    prices = np.asarray(prices, dtype=np.float32)
    returns = prices[1:] / prices[:-1] - 1
    vol = pd.Series(returns).rolling(vol_window).std().iloc[window : len(returns) - 1]
    q1, q2 = vol.quantile(1.0 / 3.0), vol.quantile(2.0 / 3.0)
    masks = [
        (vol <= q1).values,
        ((vol > q1) & (vol <= q2)).values,
        (vol > q2).values,
    ]
    return [(X[m], y[m]) for (X, y), m in zip(envs_full, masks, strict=True)]


def build_finance_envs(
    prices: np.ndarray,
    window: int = 10,
    env_mode: str = "time_spurious",
    seed: int = 0,
    label_noise: float = 0.01,
) -> list[Environment]:
    """Build ``[train_env_0, train_env_1, test_env]`` for the given ``env_mode``."""
    if env_mode == "time_spurious":
        return make_time_split_envs(prices, window, seed=seed, label_noise=label_noise)
    if env_mode == "spurious_strength":
        return make_spurious_strength_envs(prices, window, label_noise=label_noise, seed=seed)
    if env_mode == "factor":
        return make_factor_envs(prices, window)
    if env_mode == "trend_regimes":
        # Original experiments always used 1% label noise here, whatever was requested.
        return make_trend_regime_envs(prices, window, seed=seed)
    if env_mode == "volatility":
        return make_volatility_envs(prices, window, seed=seed, label_noise=label_noise)
    raise ValueError(f"Unknown env_mode {env_mode!r}; expected one of {', '.join(ENV_MODES)}")
