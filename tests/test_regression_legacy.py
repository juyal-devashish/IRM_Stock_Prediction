"""The refactored package must reproduce the original scripts' numbers exactly.

The legacy code draws from NumPy's global RNG; the package uses ``RandomState(seed)``.
Seeding the global RNG with the same value yields the identical stream, so results
should match bit-for-bit (they are asserted to 1e-9).
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from irm_stock.experiments.finance import FinanceConfig, run_finance
from irm_stock.experiments.runner import MethodConfigs
from irm_stock.types import IRMConfig, IRMGamesConfig, MLPConfig

SMALL = {"hidden_dim": 8, "l2": 1e-3}
ENV_MODES = ["time_spurious", "spurious_strength", "factor", "trend_regimes", "volatility"]


def _models():
    return MethodConfigs(
        erm=MLPConfig(lr=1e-3, steps=15, **SMALL),
        irm=IRMConfig(lr=1e-3, steps=15, penalty_anneal_iters=5, **SMALL),
        irm_games=IRMGamesConfig(lr=2.5e-4, steps=6, batch_size=64, warm_start=True, **SMALL),
    )


@pytest.mark.parametrize("env_mode", ENV_MODES)
def test_finance_matches_legacy(legacy_path, fake_prices, monkeypatch, env_mode):
    import finance_experiments_combined as legacy

    monkeypatch.setattr(legacy, "download_prices", lambda *a, **k: fake_prices)
    kw = {"env_mode": env_mode, "ticker": "x", "window": 10, "label_noise": 0.01}
    m = _models()

    np.random.seed(0)
    old_erm = legacy.train_erm(hidden_dim=8, lr=1e-3, steps=15, n_restarts=2, **kw)
    np.random.seed(0)
    old_irm = legacy.train_irm(
        hidden_dim=8, lr=1e-3, steps=15, penalty_anneal_iters=5, n_restarts=2, **kw
    )
    np.random.seed(0)
    old_games = legacy.train_irm_game(
        hidden_dim=8, lr=2.5e-4, steps=6, batch_size=64, warm_start=True, n_restarts=2, **kw
    )

    cfg = FinanceConfig(env_mode=env_mode, ticker="x", label_noise=0.01, n_restarts=2, models=m)
    res = run_finance(cfg, prices=fake_prices).summaries
    for key, old in (("erm", old_erm), ("irm", old_irm), ("irm_games", old_games)):
        np.testing.assert_allclose(res[key].train_accs, old[0], atol=1e-9, err_msg=key)
        np.testing.assert_allclose(res[key].test_accs, old[1], atol=1e-9, err_msg=key)


def _legacy_synthetic_namespace():
    src = (Path(__file__).resolve().parents[1] / "legacy" / "synthetic.py").read_text()
    # Everything after the first banner print() runs the full experiment at import time.
    ns: dict = {"__name__": "legacy_synthetic"}
    exec(compile(src.split('print("="*70)')[0], "legacy/synthetic.py", "exec"), ns)
    return ns


@pytest.mark.parametrize(
    ("ones", "hetero", "scramble"), [(False, True, True), (True, False, False)]
)
def test_synthetic_matches_legacy(ones, hetero, scramble):
    from irm_stock.experiments.synthetic import SyntheticConfig, run_configuration

    ns = _legacy_synthetic_namespace()
    old = ns["run_single_configuration"](ones, hetero, scramble, n_reps=2, n_iterations=60)
    new = run_configuration(ones, hetero, scramble, SyntheticConfig(n_reps=2, n_iterations=60))
    for legacy_name, key in (("ERM", "erm"), ("IRMGames", "irm_games"), ("IRM", "irm")):
        for err in ("causal_errors", "noncausal_errors"):
            np.testing.assert_allclose(new[key][err], old[legacy_name][err], rtol=1e-6, atol=1e-9)


def test_colored_mnist_matches_legacy(legacy_path, monkeypatch):
    from irm_stock.experiments.colored_mnist import ColoredMNISTConfig, run_colored_mnist

    rs = np.random.RandomState(7)
    images = rs.randint(0, 256, (60000, 28, 28)).astype(np.uint8)
    labels = rs.randint(0, 10, 60000).astype(np.int64)

    class _T:  # mimics a torch tensor's .numpy()
        def __init__(self, a):
            self.a = a

        def __getitem__(self, s):
            return _T(self.a[s])

        def numpy(self):
            return self.a.copy()

    fake = types.ModuleType("torchvision")
    fake.datasets = types.SimpleNamespace(
        MNIST=lambda *a, **k: types.SimpleNamespace(data=_T(images), targets=_T(labels))
    )
    monkeypatch.setitem(sys.modules, "torchvision", fake)
    import colored_mnist_exp as legacy

    kw = {"hidden_dim": 8, "n_restarts": 1}
    np.random.seed(0)
    old_erm = legacy.train_erm(lr=1e-3, steps=4, **kw)
    np.random.seed(0)
    old_irm = legacy.train_irm(lr=1e-3, steps=4, penalty_anneal_iters=2, **kw)
    np.random.seed(0)
    old_games = legacy.train_irm_game(lr=1e-3, steps=3, batch_size=64, warm_start=False, **kw)

    models = MethodConfigs(
        erm=MLPConfig(hidden_dim=8, steps=4),
        irm=IRMConfig(hidden_dim=8, steps=4, penalty_anneal_iters=2),
        irm_games=IRMGamesConfig(hidden_dim=8, lr=1e-3, steps=3, batch_size=64, warm_start=False),
    )
    res = run_colored_mnist(
        ColoredMNISTConfig(n_restarts=1, models=models), mnist=(images, labels)
    ).summaries
    for key, old in (("erm", old_erm), ("irm", old_irm), ("irm_games", old_games)):
        np.testing.assert_allclose(res[key].train_accs, old[0], atol=1e-9, err_msg=key)
        np.testing.assert_allclose(res[key].test_accs, old[1], atol=1e-9, err_msg=key)
