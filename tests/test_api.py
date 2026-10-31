import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from irm_stock.api.app import app, get_price_provider


@pytest.fixture
def client(fake_prices):
    app.dependency_overrides[get_price_provider] = lambda: lambda *a: fake_prices
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_experiments_lists_presets(client):
    body = client.get("/experiments").json()
    assert set(body["presets"]) == {"1", "2", "3"}
    assert "volatility" in body["env_modes"]


def test_run_finance_small(client):
    r = client.post(
        "/run/finance",
        json={
            "ticker": "SPY",
            "env_mode": "time_spurious",
            "steps": 10,
            "games_steps": 5,
            "hidden_dim": 8,
        },
    )
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert set(res) == {"erm", "irm", "irm_games"}
    assert 0.0 <= res["erm"]["test_mean"] <= 1.0


@pytest.mark.parametrize(
    "payload",
    [{"steps": 10_000}, {"n_restarts": 50}, {"ticker": "../etc"}, {"env_mode": "nope"}],
)
def test_run_finance_rejects_unbounded_or_invalid(client, payload):
    assert client.post("/run/finance", json=payload).status_code == 422
