"""Weekly walk-forward evaluation on data the models never saw.

Models are trained once, on the Experiment 1 (time split) environments over
2010-01-01 to 2024-01-01, then frozen. Each run scores them on every trading day
since the cutoff in two variants:

* ``shortcut_flipped``: the shortcut feature agrees with the label only 10% of the
  time, like the report's test environment. A model that relies on it falls below 50%.
* ``no_shortcut``: the shortcut is a coin flip (50%), so only the real return signal
  can help. This measures whether there is any genuine predictive edge.

The up-day base rate is reported too: "always predict up" scores exactly that.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from irm_stock.core import ensemble_logits, forward, mean_accuracy
from irm_stock.data.finance import (
    build_finance_envs,
    classify_trend,
    download_prices,
    fetch_close_series,
    make_finance_environment,
    sideways_threshold,
    trend_series,
)
from irm_stock.experiments.runner import MethodConfigs, train_method
from irm_stock.methods import METHODS

logger = logging.getLogger(__name__)

TRAIN_START = "2010-01-01"
CUTOFF = "2024-01-01"
DEFAULT_TICKERS = ("SPY", "AAPL", "WMT", "VOO")
VARIANTS = {"shortcut_flipped": 0.9, "no_shortcut": 0.5}  # value = e (disagreement prob.)
VOL_WINDOW = 20
#: Extra days fetched before the cutoff so moving averages are defined at the cutoff.
LOOKBACK_DAYS = 420

Models = dict[str, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class FrozenSpec:
    ticker: str
    window: int = 10
    seed: int = 0

    @property
    def filename(self) -> str:
        return f"{self.ticker.upper()}_time_spurious_w{self.window}_s{self.seed}.npz"


# ------------------------------------------------------------------------- models


def train_frozen(prices_train: np.ndarray, spec: FrozenSpec) -> Models:
    """Train one model per method on the time-split environments (no label noise)."""
    envs = build_finance_envs(
        prices_train, window=spec.window, env_mode="time_spurious", seed=spec.seed, label_noise=0.0
    )
    configs = MethodConfigs()
    models: Models = {}
    for method in METHODS:
        result = train_method(method, envs, configs, np.random.RandomState(spec.seed))
        models[method] = (result.W, result.V)
    return models


def save_models(models: Models, path: Path, spec: FrozenSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{m}_{part}": arr for m, (W, V) in models.items() for part, arr in (("W", W), ("V", V))
    }
    meta = {"ticker": spec.ticker, "window": spec.window, "seed": spec.seed, "train_end": CUTOFF}
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)


def load_models(path: Path) -> Models:
    with np.load(path) as data:
        return {m: (data[f"{m}_W"], data[f"{m}_V"]) for m in METHODS if f"{m}_W" in data}


def logits(W: np.ndarray, V: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Logits of a single MLP (2-D ``W``) or an IRM-Games ensemble (3-D ``W``)."""
    return ensemble_logits(X, W, V) if W.ndim == 3 else forward(X, W, V)[2][:, 0]


# --------------------------------------------------------------------- evaluation


def sample_label_dates(dates: pd.DatetimeIndex, window: int) -> pd.DatetimeIndex:
    """Date of the label day for each sample built by ``make_finance_environment``.

    Sample ``i`` uses returns ``i .. i+window-1`` as features and the return on price
    day ``i + window + 2`` as its label (the original construction skips one day).
    """
    n = len(dates) - window - 2
    return dates[window + 2 : window + 2 + n]


def evaluate(
    models: Models, series: pd.Series, spec: FrozenSpec, cutoff: str = CUTOFF
) -> dict[str, Any]:
    """Score frozen models on samples whose label day is on or after ``cutoff``."""
    prices = series.to_numpy(dtype=np.float32)
    label_dates = sample_label_dates(pd.DatetimeIndex(series.index), spec.window)
    mask = label_dates >= pd.Timestamp(cutoff)
    if not mask.any():
        raise ValueError(f"no trading days after {cutoff} for {spec.ticker}")
    dates = label_dates[mask]
    months = dates.strftime("%Y-%m")

    out: dict[str, Any] = {"n_days": int(mask.sum()), "variants": {}, "monthly": []}
    per_variant_correct: dict[str, dict[str, np.ndarray]] = {}
    for variant, e in VARIANTS.items():
        X, y = make_finance_environment(prices, spec.window, e=e, seed=spec.seed, label_noise=0.0)
        X, y = X[mask], y[mask]
        out["base_rate"] = float(y.mean())
        labels = y  # identical across variants (no label noise)
        out["variants"][variant] = {
            m: float(mean_accuracy(logits(W, V, X), y)) for m, (W, V) in models.items()
        }
        per_variant_correct[variant] = {
            m: ((logits(W, V, X) > 0).astype(np.float32) == y) for m, (W, V) in models.items()
        }

    for month in sorted(set(months)):
        sel = np.asarray(months == month)
        row: dict[str, Any] = {
            "month": month,
            "n_days": int(sel.sum()),
            "base_rate": float(labels[sel].mean()),
        }
        for variant, correct in per_variant_correct.items():
            row[variant] = {m: float(c[sel].mean()) for m, c in correct.items()}
        out["monthly"].append(row)
    return out


def current_regime(series: pd.Series, prices_train: np.ndarray) -> dict[str, Any]:
    """Market regime on the last available day, with thresholds from the training period."""
    prices = series.to_numpy(dtype=np.float32)
    gap = trend_series(prices)[-1]
    trend = (
        "unknown"
        if np.isnan(gap)
        else classify_trend(gap, sideways_threshold(trend_series(prices_train)))
    )

    train_returns = prices_train[1:] / prices_train[:-1] - 1
    train_vol = pd.Series(train_returns).rolling(VOL_WINDOW).std().dropna()
    q1, q2 = train_vol.quantile(1 / 3), train_vol.quantile(2 / 3)
    returns = prices[1:] / prices[:-1] - 1
    vol = float(pd.Series(returns).rolling(VOL_WINDOW).std().iloc[-1])
    vol_label = "low" if vol <= q1 else "mid" if vol <= q2 else "high"
    return {"trend": trend, "volatility": vol_label, "vol_20d": vol}


# ------------------------------------------------------------------------ history


def iso_week(day: dt.date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def append_snapshot(history_path: Path, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Add ``snapshot`` to the history, replacing any snapshot from the same ISO week."""
    history: list[dict[str, Any]] = []
    if history_path.exists():
        history = json.loads(history_path.read_text())
    history = [s for s in history if s.get("week") != snapshot["week"]]
    history.append(snapshot)
    history.sort(key=lambda s: s["week"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=1) + "\n")
    return history


# ------------------------------------------------------------------------- driver


def _lookback_start(cutoff: str) -> str:
    return (dt.date.fromisoformat(cutoff) - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()


def run_ticker(
    spec: FrozenSpec,
    models_dir: Path,
    retrain: bool = False,
    prices_train: np.ndarray | None = None,
    series: pd.Series | None = None,
) -> dict[str, Any]:
    """Train (or load) the frozen models for one ticker and evaluate them since the cutoff."""
    if prices_train is None:
        prices_train = download_prices(spec.ticker, TRAIN_START, CUTOFF)
    path = models_dir / spec.filename
    if path.exists() and not retrain:
        models = load_models(path)
    else:
        logger.info("Training frozen models for %s", spec.ticker)
        models = train_frozen(prices_train, spec)
        save_models(models, path, spec)

    if series is None:
        tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()
        series = fetch_close_series(spec.ticker, _lookback_start(CUTOFF), tomorrow)
    result = evaluate(models, series, spec)
    result["as_of"] = pd.Timestamp(series.index[-1]).date().isoformat()
    result["regime"] = current_regime(series, prices_train)
    return result


def run_walk_forward(
    tickers: tuple[str, ...] | list[str],
    data_dir: Path,
    window: int = 10,
    seed: int = 0,
    retrain: bool = False,
    today: dt.date | None = None,
) -> dict[str, Any]:
    """Evaluate every ticker and append one snapshot to ``<data_dir>/history.json``.

    A ticker that fails (e.g. a download error) is recorded with its error; the run
    fails only if every ticker fails.
    """
    today = today or dt.date.today()
    snapshot: dict[str, Any] = {
        "week": iso_week(today),
        "run_date": today.isoformat(),
        "train": {"start": TRAIN_START, "end": CUTOFF, "env_mode": "time_spurious"},
        "tickers": {},
        "errors": {},
    }
    for ticker in tickers:
        spec = FrozenSpec(ticker.upper(), window, seed)
        try:
            snapshot["tickers"][spec.ticker] = run_ticker(spec, data_dir / "models", retrain)
        except Exception as exc:  # keep going; one bad download should not lose the week
            logger.exception("Walk-forward failed for %s", spec.ticker)
            snapshot["errors"][spec.ticker] = f"{type(exc).__name__}: {exc}"
    if not snapshot["tickers"]:
        raise RuntimeError(f"walk-forward failed for every ticker: {snapshot['errors']}")
    append_snapshot(data_dir / "history.json", snapshot)
    return snapshot
