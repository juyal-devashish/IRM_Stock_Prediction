# IRM Stock Prediction

Can invariant learning ignore spurious signals in financial data? This project compares
**ERM**, **IRMv1** and **IRM-Games** (pure NumPy, hand-written gradients) on:

- **Finance** — predict next-day return sign from stock data, with a spurious feature whose
  relationship to the label shifts across time / market regimes / volatility regimes.
- **ColoredMNIST** — the standard IRM benchmark.
- **Synthetic SEM** — Figure 4 of the IRM paper, with ICP replaced by IRM-Games.

Originally an ECE 553 final project.

## Quick start

```bash
uv sync --extra api            # add --extra mnist for ColoredMNIST
uv run irm-stock finance --experiment 2 --n-restarts 1 --steps 100 --games-steps 100
uv run irm-stock synthetic --configs FOU,PES --n-reps 2 --n-iterations 2000
uv run irm-stock serve         # demo API at http://127.0.0.1:8000/docs
```

Runs write `results.json`, a plot and `run.log` to `outputs/<experiment>/<timestamp>_<slug>/`.

## Layout

| Path | Contents |
|---|---|
| `src/irm_stock/` | Package: `core`, `methods`, `data`, `experiments`, `plotting`, `cli`, `api` |
| `tests/` | Unit tests plus bit-for-bit regression tests against `legacy/` |
| `legacy/` | Original research scripts (unmodified, kept as the numerical baseline) |
| `results/` | Curated figures, logs and saved arrays from the reported runs |
| `notebooks/` | Exploration notebooks |
| `outputs/` | New run output (gitignored) |

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

## API

`GET /health`, `GET /experiments`, `POST /run/finance` (bounded: ≤3 restarts, ≤300 steps,
one job at a time), static figures under `/figures`.
