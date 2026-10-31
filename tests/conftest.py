import sys
from pathlib import Path

import numpy as np
import pytest

LEGACY_DIR = Path(__file__).resolve().parents[1] / "legacy"


@pytest.fixture(scope="session")
def legacy_path():
    """Make the unmodified research scripts importable as top-level modules."""
    sys.path.insert(0, str(LEGACY_DIR))
    yield LEGACY_DIR
    sys.path.remove(str(LEGACY_DIR))


@pytest.fixture
def fake_prices():
    """~3 years of deterministic random-walk prices (no network needed)."""
    rng = np.random.RandomState(123)
    return (100 * np.exp(np.cumsum(rng.normal(3e-4, 0.012, 900)))).astype(np.float32)
