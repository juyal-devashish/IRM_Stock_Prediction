"""Command-line interface: ``irm-stock {finance,colored-mnist,synthetic,serve}``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from irm_stock.data.finance import ENV_MODES
from irm_stock.experiments.runner import MethodConfigs
from irm_stock.methods import METHODS
from irm_stock.types import IRMConfig, IRMGamesConfig, MLPConfig

logger = logging.getLogger("irm_stock")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def _attach_run_log(run_dir: Path, lines: str) -> None:
    (run_dir / "run.log").write_text(lines)


class _Capture(logging.Handler):
    """Collects log output so it can be written into the run directory."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="Base random seed.")
    p.add_argument(
        "--methods",
        default=",".join(METHODS),
        help=f"Comma-separated subset of: {', '.join(METHODS)}.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Root for run directories."
    )


def _add_mlp(p: argparse.ArgumentParser, games_lr: float, games_warm_start: bool) -> None:
    g = p.add_argument_group("model hyper-parameters")
    g.add_argument("--hidden-dim", type=int, default=256)
    g.add_argument("--l2", type=float, default=1e-3)
    g.add_argument("--lr", type=float, default=1e-3, help="Learning rate for ERM and IRM.")
    g.add_argument("--steps", type=int, default=501, help="Training steps for ERM and IRM.")
    g.add_argument("--penalty-weight", type=float, default=10_000.0)
    g.add_argument("--penalty-anneal-iters", type=int, default=100)
    g.add_argument("--games-lr", type=float, default=games_lr, help="IRM-Games learning rate.")
    g.add_argument("--games-steps", type=int, default=500, help="IRM-Games rounds.")
    g.add_argument("--batch-size", type=int, default=256, help="IRM-Games minibatch size.")
    g.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=games_warm_start,
        help="Pre-train each IRM-Games player with ERM on its own environment.",
    )


def _model_configs(args: argparse.Namespace) -> MethodConfigs:
    common = {"hidden_dim": args.hidden_dim, "l2": args.l2}
    return MethodConfigs(
        erm=MLPConfig(lr=args.lr, steps=args.steps, **common),
        irm=IRMConfig(
            lr=args.lr,
            steps=args.steps,
            penalty_weight=args.penalty_weight,
            penalty_anneal_iters=args.penalty_anneal_iters,
            **common,
        ),
        irm_games=IRMGamesConfig(
            lr=args.games_lr,
            steps=args.games_steps,
            batch_size=args.batch_size,
            warm_start=args.warm_start,
            **common,
        ),
    )


def _methods(arg: str) -> list[str]:
    return [m for m in arg.split(",") if m.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irm-stock", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    fin = sub.add_parser("finance", help="Stock-market environments (downloads from yfinance).")
    fin.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Preset: 1=time split/AAPL, 2=trend regimes/SPY, 3=volatility/WMT.",
    )
    fin.add_argument("--env-mode", choices=list(ENV_MODES), help="Override the preset env mode.")
    fin.add_argument("--ticker", help="Override the preset ticker, e.g. SPY.")
    fin.add_argument("--label-noise", type=float, help="Override the preset label-noise rate.")
    fin.add_argument("--start", default="2010-01-01")
    fin.add_argument("--end", default="2024-01-01")
    fin.add_argument("--window", type=int, default=10, help="Days of returns per sample.")
    _add_common(fin)
    _add_mlp(fin, games_lr=2.5e-4, games_warm_start=True)

    mn = sub.add_parser("colored-mnist", help="ColoredMNIST benchmark (needs the mnist extra).")
    mn.add_argument("--data-dir", help="MNIST download/cache dir (default ~/datasets/mnist).")
    _add_common(mn)
    _add_mlp(mn, games_lr=1e-3, games_warm_start=False)

    syn = sub.add_parser("synthetic", help="Figure 4 of the IRM paper (linear SEM).")
    syn.add_argument("--configs", default="all", help="Comma-separated, e.g. FOU,PES.")
    syn.add_argument("--n-reps", type=int, default=10)
    syn.add_argument("--n-iterations", type=int, default=100_000)
    syn.add_argument("--seed", type=int, default=0)
    syn.add_argument("--output-dir", type=Path, default=Path("outputs"))

    wf = sub.add_parser(
        "walk-forward",
        help="Score frozen (pre-2024) models on all data since 2024; append to a history.",
    )
    wf.add_argument(
        "--tickers",
        default="SPY,AAPL,NVDA,MSFT,GOOGL,AMZN,WMT,VOO",
        help="Comma-separated tickers.",
    )
    wf.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Holds models/, cache/ and history.json (the `data` branch in CI).",
    )
    wf.add_argument("--window", type=int, default=10)
    wf.add_argument("--seed", type=int, default=0)
    wf.add_argument("--retrain", action="store_true", help="Retrain even if saved models exist.")

    srv = sub.add_parser("serve", help="Run the demo API server (needs the api extra).")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--reload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    capture = _Capture()
    logger.addHandler(capture)

    if args.command == "finance":
        from irm_stock.experiments.finance import (
            PRESETS,
            FinanceConfig,
            run_finance,
            save_finance_result,
        )

        config = FinanceConfig.from_preset(
            args.experiment,
            env_mode=args.env_mode,
            ticker=args.ticker,
            label_noise=args.label_noise,
            start=args.start,
            end=args.end,
            window=args.window,
            n_restarts=args.n_restarts,
            seed=args.seed,
            methods=_methods(args.methods),
            models=_model_configs(args),
        )
        logger.info(PRESETS[args.experiment].name)
        logger.info(
            "env_mode=%s | ticker=%s | window=%d | label_noise=%s",
            config.env_mode,
            config.ticker,
            config.window,
            config.label_noise,
        )
        logger.info("date range: %s to %s", config.start, config.end)
        run_dir = save_finance_result(run_finance(config), args.output_dir, f"exp{args.experiment}")

    elif args.command == "colored-mnist":
        from irm_stock.experiments.colored_mnist import (
            ColoredMNISTConfig,
            run_colored_mnist,
            save_colored_mnist_result,
        )

        config = ColoredMNISTConfig(
            n_restarts=args.n_restarts,
            seed=args.seed,
            methods=_methods(args.methods),
            data_dir=args.data_dir,
            models=_model_configs(args),
        )
        logger.info("Coloured MNIST: ERM vs IRM vs IRM-Games")
        run_dir = save_colored_mnist_result(run_colored_mnist(config), args.output_dir)

    elif args.command == "synthetic":
        from irm_stock.experiments.synthetic import (
            ALL_CONFIG_NAMES,
            SyntheticConfig,
            run_synthetic,
            save_synthetic_result,
        )

        configs = ALL_CONFIG_NAMES if args.configs == "all" else args.configs.split(",")
        config = SyntheticConfig(
            configs=configs, n_reps=args.n_reps, n_iterations=args.n_iterations, seed=args.seed
        )
        run_dir = save_synthetic_result(run_synthetic(config), config, args.output_dir)

    elif args.command == "walk-forward":
        import os

        from irm_stock.pipeline.walk_forward import run_walk_forward

        os.environ.setdefault("IRM_STOCK_CACHE_DIR", str(args.data_dir / "cache"))
        snap = run_walk_forward(
            [t for t in args.tickers.split(",") if t.strip()],
            args.data_dir,
            window=args.window,
            seed=args.seed,
            retrain=args.retrain,
        )
        for ticker, res in snap["tickers"].items():
            v = res["variants"]
            logger.info(
                "%s (%d days to %s, regime %s/%s): flipped %s | no shortcut %s | up-days %.1f%%",
                ticker,
                res["n_days"],
                res["as_of"],
                res["regime"]["trend"],
                res["regime"]["volatility"],
                ", ".join(f"{m} {a:.1%}" for m, a in v["shortcut_flipped"].items()),
                ", ".join(f"{m} {a:.1%}" for m, a in v["no_shortcut"].items()),
                100 * res["base_rate"],
            )
        for ticker, err in snap["errors"].items():
            logger.error("%s failed: %s", ticker, err)
        return 0

    elif args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            logger.error("The API server needs the api extra: uv sync --extra api")
            return 1
        uvicorn.run("irm_stock.api.app:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    _attach_run_log(run_dir, capture.text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
