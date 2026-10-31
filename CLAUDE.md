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

## Gotchas
- Refactors must keep `tests/test_regression_legacy.py` passing. Numerical changes (even "bug fixes") need an explicit decision from the user and re-baselined results.
- Reproducibility: pass explicit `np.random.RandomState`; don't use the global `np.random`. Random draw order matters (W is drawn before V; methods consume the stream in a fixed order).
- Intentional legacy quirks preserved: `linear.fit_irm_games` uses the global iteration count for Adam bias correction; `trend_regimes` env ignores `label_noise` (always 1%); IRM penalty gradient is rescaled by `1/penalty_weight` after annealing.
- Finance data needs internet (yfinance); past date ranges are cached in `~/.cache/irm_stock` (override `IRM_STOCK_CACHE_DIR`). Tests use synthetic prices, never the network.
- Long runs: `synthetic` at defaults takes hours; ColoredMNIST tens of minutes. Use small `--n-restarts/--steps/--n-iterations` when testing.
- API endpoints must bound request size (see `api/schemas.py` limits); training is CPU-bound NumPy, one job at a time.

## Git
- Commit after each substantial change, with a descriptive message. Work on a feature branch, not directly on `main`.
