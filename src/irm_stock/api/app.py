"""Demo API: describe the experiments, serve curated results, run small jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from irm_stock import __version__
from irm_stock.api.schemas import FinanceRunRequest, FinanceRunResponse
from irm_stock.data.finance import ENV_MODES, download_prices
from irm_stock.experiments.finance import PRESETS, FinanceConfig, run_finance
from irm_stock.experiments.runner import MethodConfigs
from irm_stock.methods import METHOD_LABELS
from irm_stock.types import IRMConfig, IRMGamesConfig, MLPConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO_ROOT / "results" / "figures"

#: Only one training job at a time: they are CPU-bound NumPy and the host is small.
_run_lock = threading.Semaphore(1)

PriceProvider = Callable[[str, str, str], np.ndarray]


def get_price_provider() -> PriceProvider:
    """Dependency so tests can swap in offline prices."""
    return download_prices


def create_app(cors_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="IRM Stock Prediction demo", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:3000", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/experiments")
    def experiments() -> dict:
        return {
            "presets": {
                str(k): {
                    "name": p.name,
                    "env_mode": p.env_mode,
                    "ticker": p.ticker,
                    "label_noise": p.label_noise,
                }
                for k, p in PRESETS.items()
            },
            "env_modes": ENV_MODES,
            "methods": METHOD_LABELS,
        }

    @app.post("/run/finance", response_model=FinanceRunResponse)
    def run_finance_endpoint(
        req: FinanceRunRequest,
        prices_for: Annotated[PriceProvider, Depends(get_price_provider)],
    ) -> FinanceRunResponse:
        if not _run_lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Another run is in progress; retry soon.")
        try:
            try:
                prices = prices_for(req.ticker, req.start, req.end)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            common = {"hidden_dim": req.hidden_dim}
            config = FinanceConfig(
                env_mode=req.env_mode,
                ticker=req.ticker,
                start=req.start,
                end=req.end,
                window=req.window,
                label_noise=req.label_noise,
                n_restarts=req.n_restarts,
                seed=req.seed,
                methods=req.methods,
                models=MethodConfigs(
                    erm=MLPConfig(steps=req.steps, **common),
                    irm=IRMConfig(
                        steps=req.steps,
                        penalty_anneal_iters=max(1, req.steps // 5),
                        **common,
                    ),
                    irm_games=IRMGamesConfig(steps=req.games_steps, **common),
                ),
            )
            try:
                result = run_finance(config, prices=prices)
            except ValueError as exc:  # e.g. too little data for a regime
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return FinanceRunResponse(
                request=req, results={k: s.to_dict() for k, s in result.summaries.items()}
            )
        finally:
            _run_lock.release()

    if FIGURES_DIR.is_dir():
        app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")
    return app


app = create_app()
