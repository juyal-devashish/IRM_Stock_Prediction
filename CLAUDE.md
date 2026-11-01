# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Research code comparing ERM, IRMv1 and IRM-Games (pure NumPy, hand-written gradients) on finance, ColoredMNIST and synthetic-SEM tasks. A FastAPI demo backend is planned/in progress (`src/irm_stock/api/`); a separate frontend will consume it.

## Commands (uv; Python 3.12)
- Setup: `uv sync --extra api` (add `--extra mnist` for ColoredMNIST; pulls torch via torchvision)
- Check everything: `uv run ruff check . && uv run ruff format --check . && uv run pytest`
- Single test: `uv run pytest tests/test_core.py -k gradients`
- Experiments: `uv run irm-stock {finance,colored-mnist,synthetic} --help`; API: `uv run irm-stock serve`

## Layout
- `src/irm_stock/`: `core/` (MLP, losses, Adam), `methods/` (ERM/IRM/IRM-Games), `data/`, `experiments/`, `plotting.py`, `cli.py`, `api/`
- `legacy/`: original research scripts. **Never edit.** `tests/test_regression_legacy.py` compares the package against them bit-for-bit.
- `results/`: curated, committed figures/logs for reports and the website. `outputs/`: new run directories (gitignored). Never write run output to the repo root.

## Dashboard
- `frontend/` is Vite + React + TS (no chart library; charts are hand-written SVG). Data comes from `scripts/build_dashboard_data.py`, which parses `results/`; `frontend/public/{data,figures}` are generated and gitignored.
- Check: `uv run python scripts/build_dashboard_data.py && cd frontend && npm run build`. CI (`.github/workflows/ci.yml`) runs both Python and frontend jobs; `pages.yml` deploys `main` to GitHub Pages with `VITE_BASE=/<repo>/`.
- Extra report tickers: `uv run python scripts/run_report_experiments.py --tickers ... --skip-existing` writes `results/runs/finance/exp<N>_<TICKER>.json`; the builder uses them only for tickers with no original log (marked * on the page). Add new tickers to `pipeline.yml` too.
- `frontend/src/lab/` is a TypeScript port of `core/` + `methods/`. After changing either side, run `uv run python scripts/export_lab_fixtures.py` and `cd frontend && npm test`; never loosen the parity tolerances to make them pass.
- Colours: ERM blue, IRM orange, IRM-Games aqua (validated categorical slots 1-3, light and dark). Keep a method's colour fixed across every chart.

## Walk-forward pipeline
- `uv run irm-stock walk-forward --data-dir data` trains frozen pre-2024 models once and scores them on all data since 2024. `.github/workflows/pipeline.yml` runs it weekly against a checkout of the `data` branch (models, price cache, `history.json`); never commit `data/` to main.
- IRM's post-2024 accuracy equals the up-day base rate (it predicts "up"): keep the base rate next to it in any UI.

## Gotchas
- Refactors must keep `tests/test_regression_legacy.py` passing. Numerical changes (even "bug fixes") need an explicit decision from the user and re-baselined results.
- Reproducibility: pass explicit `np.random.RandomState`; don't use the global `np.random`. Random draw order matters (W is drawn before V; methods consume the stream in a fixed order).
- Intentional legacy quirks preserved: `linear.fit_irm_games` uses the global iteration count for Adam bias correction; `trend_regimes` env ignores `label_noise` (always 1%); IRM penalty gradient is rescaled by `1/penalty_weight` after annealing.
- Finance data needs internet (yfinance); past date ranges are cached in `~/.cache/irm_stock` (override `IRM_STOCK_CACHE_DIR`). Tests use synthetic prices, never the network.
- Long runs: `synthetic` at defaults takes hours; ColoredMNIST tens of minutes. Use small `--n-restarts/--steps/--n-iterations` when testing.
- API endpoints must bound request size (see `api/schemas.py` limits); training is CPU-bound NumPy, one job at a time.

## Git
- Commit after each substantial change, with a descriptive message. Work on a feature branch, not directly on `main`.
