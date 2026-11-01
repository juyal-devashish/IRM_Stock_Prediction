# IRM Stock Prediction

Can invariant learning ignore spurious signals in financial data? This project compares
**ERM**, **IRMv1** and **IRM-Games** (pure NumPy, hand-written gradients) on:

- **Finance**: predict next-day return sign from stock data, with a spurious feature whose
  relationship to the label shifts across time, market regimes and volatility regimes.
  Tickers: the S&P 500 (SPY), AAPL, NVDA, MSFT, GOOGL, AMZN, WMT and VOO.
- **ColoredMNIST**: the standard IRM benchmark.
- **Synthetic SEM**: Figure 4 of the IRM paper, with ICP replaced by IRM-Games.

Originally an ECE 553 final project.

**Live site:** https://juyal-devashish.github.io/IRM_Stock_Prediction/

- A train-it-yourself lab: the three methods train in your browser on toy data, with a shortcut dial.
- Report results for every benchmark and ticker.
- "Since 2024": models frozen on 2010–2023 data, scored every week on all trading days since.

## Quick start

```bash
uv sync --extra api            # add --extra mnist for ColoredMNIST
uv run irm-stock finance --experiment 2 --n-restarts 1 --steps 100 --games-steps 100
uv run irm-stock synthetic --configs FOU,PES --n-reps 2 --n-iterations 2000
uv run irm-stock walk-forward --tickers SPY --data-dir data
uv run irm-stock serve         # FastAPI demo at http://127.0.0.1:8000/docs (not deployed)
```

Experiment runs write `results.json`, a plot and `run.log` to
`outputs/<experiment>/<timestamp>_<slug>/` (gitignored).

## Dashboard

```bash
uv run python scripts/build_dashboard_data.py   # results/ -> frontend/public/{data,figures}
cd frontend && npm ci && npm run dev
```

- `frontend/public/{data,figures}` are generated and gitignored.
- Finance results come from the original report logs in `results/logs/finance/` (only those whose filename names
  the ticker), plus `results/runs/finance/*.json` for tickers without a log. Regenerate or add tickers with
  `uv run python scripts/run_report_experiments.py --tickers ... --skip-existing`.
- `frontend/src/lab/` is a TypeScript port of the training code. `npm test` checks it against fixtures from
  `scripts/export_lab_fixtures.py`.

## Weekly walk-forward

`.github/workflows/pipeline.yml` runs every Monday at 06:00 UTC, and you can also start it from the Actions tab.
- It runs `irm-stock walk-forward` against the `data` branch, which holds the frozen models, the price cache and
  `history.json`.
- It then redeploys the site.
- Tickers are listed in the workflow file.

## Layout

| Path | Contents |
|---|---|
| `src/irm_stock/` | Package: `core`, `methods`, `data`, `experiments`, `pipeline`, `plotting`, `cli`, `api` |
| `tests/` | Unit tests plus bit-for-bit regression tests against `legacy/` |
| `legacy/` | Original research scripts (unmodified, kept as the numerical baseline) |
| `results/` | Curated figures, logs, saved arrays and extra report runs |
| `notebooks/` | Exploration notebooks |
| `frontend/` | Dashboard and in-browser lab (Vite + React + TypeScript) |
| `scripts/` | Dashboard data builder, report-run and lab-fixture generators |
| `.github/workflows/` | `ci.yml` (lint, tests, parity, build), `pages.yml` (deploy), `pipeline.yml` (weekly) |

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
cd frontend && npm test && npm run build
```

## API

`GET /health`, `GET /experiments`, `POST /run/finance` (bounded: ≤3 restarts, ≤300 steps,
one job at a time), static figures under `/figures`.
