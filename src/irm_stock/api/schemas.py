"""Request/response models. Limits keep a public demo from running unbounded jobs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from irm_stock.data.finance import ENV_MODES

EnvMode = Literal[tuple(ENV_MODES)]  # type: ignore[valid-type]
Method = Literal["erm", "irm", "irm_games"]

MAX_RESTARTS = 3
MAX_STEPS = 300
MAX_HIDDEN = 128


class FinanceRunRequest(BaseModel):
    ticker: str = Field("SPY", pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")
    env_mode: EnvMode = "trend_regimes"
    methods: list[Method] = Field(default_factory=lambda: ["erm", "irm", "irm_games"], min_length=1)
    start: str = Field("2010-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field("2024-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")
    window: int = Field(10, ge=2, le=60)
    label_noise: float = Field(0.01, ge=0.0, le=0.5)
    n_restarts: int = Field(1, ge=1, le=MAX_RESTARTS)
    steps: int = Field(200, ge=1, le=MAX_STEPS, description="Steps for ERM / IRM.")
    games_steps: int = Field(200, ge=1, le=MAX_STEPS, description="Rounds for IRM-Games.")
    hidden_dim: int = Field(64, ge=1, le=MAX_HIDDEN)
    seed: int = Field(0, ge=0, le=2**31 - 1)


class MethodResult(BaseModel):
    method: str
    train_accs: list[float]
    test_accs: list[float]
    train_mean: float
    train_std: float
    test_mean: float
    test_std: float


class FinanceRunResponse(BaseModel):
    request: FinanceRunRequest
    results: dict[str, MethodResult]
