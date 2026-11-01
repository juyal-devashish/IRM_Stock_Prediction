import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from irm_stock.data.finance import make_finance_environment
from irm_stock.experiments.runner import MethodConfigs
from irm_stock.pipeline import walk_forward as wf
from irm_stock.types import IRMConfig, IRMGamesConfig, MLPConfig


@pytest.fixture
def small_models(monkeypatch):
    """Tiny, fast model configs so training takes milliseconds."""
    small = MethodConfigs(
        erm=MLPConfig(hidden_dim=8, steps=20),
        irm=IRMConfig(hidden_dim=8, steps=20, penalty_anneal_iters=5),
        irm_games=IRMGamesConfig(hidden_dim=8, steps=10, batch_size=64),
    )
    monkeypatch.setattr(wf, "MethodConfigs", lambda: small)


@pytest.fixture
def series():
    """Business-day prices spanning the cutoff (about 2 years after it)."""
    rng = np.random.RandomState(9)
    idx = pd.bdate_range("2023-01-02", "2025-12-31")
    vals = 100 * np.exp(np.cumsum(rng.normal(3e-4, 0.012, len(idx))))
    return pd.Series(vals.astype(np.float32), index=idx)


def test_label_dates_match_environment_construction(series):
    prices = series.to_numpy()
    window = 10
    _, y = make_finance_environment(prices, window, e=0.5, seed=0, label_noise=0.0)
    dates = wf.sample_label_dates(series.index, window)
    assert len(dates) == len(y)
    # Each label is "did the price rise on its label date?"
    pos = series.index.get_indexer(dates)
    np.testing.assert_array_equal(y, (prices[pos] > prices[pos - 1]).astype(np.float32))


def test_run_ticker_trains_saves_and_reuses(tmp_path, fake_prices, series, small_models):
    spec = wf.FrozenSpec("TEST")
    first = wf.run_ticker(spec, tmp_path, prices_train=fake_prices, series=series)
    assert (tmp_path / spec.filename).exists()

    n_expected = int((wf.sample_label_dates(series.index, 10) >= "2024-01-01").sum())
    assert first["n_days"] == n_expected
    assert 0 < first["base_rate"] < 1
    for variant in wf.VARIANTS:
        accs = first["variants"][variant]
        assert set(accs) == {"erm", "irm", "irm_games"}
        assert all(0.0 <= a <= 1.0 for a in accs.values())
    assert sum(m["n_days"] for m in first["monthly"]) == first["n_days"]
    weighted = sum(m["base_rate"] * m["n_days"] for m in first["monthly"]) / first["n_days"]
    assert weighted == pytest.approx(first["base_rate"])
    assert first["regime"]["trend"] in {"bull", "bear", "side", "unknown"}
    assert first["regime"]["volatility"] in {"low", "mid", "high"}

    # Second run loads the saved models and gives identical results.
    second = wf.run_ticker(spec, tmp_path, prices_train=fake_prices, series=series)
    assert second["variants"] == first["variants"]


def test_save_load_roundtrip(tmp_path, fake_prices, small_models):
    spec = wf.FrozenSpec("TEST")
    models = wf.train_frozen(fake_prices, spec)
    wf.save_models(models, tmp_path / "m.npz", spec)
    loaded = wf.load_models(tmp_path / "m.npz")
    X = np.random.RandomState(0).randn(5, 11).astype(np.float32)
    for m, (W, V) in models.items():
        np.testing.assert_array_equal(wf.logits(W, V, X), wf.logits(*loaded[m], X))


def test_append_snapshot_replaces_same_week(tmp_path):
    path = tmp_path / "history.json"
    wf.append_snapshot(path, {"week": "2026-W44", "v": 1})
    wf.append_snapshot(path, {"week": "2026-W43", "v": 0})
    wf.append_snapshot(path, {"week": "2026-W44", "v": 2})
    hist = json.loads(path.read_text())
    assert [s["week"] for s in hist] == ["2026-W43", "2026-W44"]
    assert hist[-1]["v"] == 2


def test_run_walk_forward_records_errors(tmp_path, monkeypatch):
    def boom(spec, *a, **k):
        if spec.ticker == "BAD":
            raise ValueError("no data")
        return {"n_days": 1}

    monkeypatch.setattr(wf, "run_ticker", boom)
    snap = wf.run_walk_forward(["GOOD", "BAD"], tmp_path, today=dt.date(2026, 10, 26))
    assert snap["week"] == "2026-W44"
    assert "GOOD" in snap["tickers"] and "BAD" in snap["errors"]

    with pytest.raises(RuntimeError):
        wf.run_walk_forward(["BAD"], tmp_path)
