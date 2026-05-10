#!/usr/bin/env python3
"""
Run the full TSMOM backtest pipeline.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --config config.yaml
    python scripts/run_backtest.py --symbol HPG   # single-stock mode
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsmom.data import load_data, load_benchmark_csv
from tsmom.volatility import compute_weekly_volatility
from tsmom.regression import run_tsmom_regressions, run_sign_regressions
from tsmom.backtest import backtest_single, backtest_universe
from tsmom.metrics import calculate_metrics
from tsmom.plotting import (
    plot_tstats_by_lag,
    plot_single_backtest,
    plot_portfolio_cumret,
    plot_distribution,
    plot_tsmom_vs_benchmark,
    plot_volatility_price,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="TSMOM Backtest Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--symbol", default=None, help="Run single-stock backtest")
    parser.add_argument("--no-regression", action="store_true", help="Skip regressions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg["output"].get("dpi", 200)

    strat = cfg["strategy"]
    vol_target = strat["vol_target"]
    lookback = strat["lookback_window"]
    commission = strat["commission"]
    margin_cap = strat["margin_cap"]
    ewm_com = strat["ewm_com"]

    # ── Load data ────────────────────────────────
    logger.info("Loading price data…")
    daily_prices = load_data(cfg)
    logger.info("Loaded %d days × %d symbols", len(daily_prices), len(daily_prices.columns))

    # ── Weekly data ──────────────────────────────
    weekly_prices = daily_prices.resample("W").last().ffill()
    weekly_returns = weekly_prices.pct_change().dropna()

    # ── Volatility ───────────────────────────────
    logger.info("Computing ex-ante volatility…")
    weekly_vol = compute_weekly_volatility(daily_prices, com=ewm_com)

    # ── Regression evidence ──────────────────────
    if not args.no_regression:
        max_lag = cfg["regression"]["max_lag"]
        logger.info("Running TSMOM regressions (max_lag=%d)…", max_lag)
        tsmom_reg = run_tsmom_regressions(weekly_returns, weekly_vol, max_lag=max_lag)
        sign_reg = run_sign_regressions(weekly_returns, weekly_vol, max_lag=max_lag)

        plot_tstats_by_lag(
            tsmom_reg["tstat"],
            title="TSMOM Regression — T-stat by Lag",
            save_path=out_dir / "tstat_tsmom.png",
            dpi=dpi,
        )
        plot_tstats_by_lag(
            sign_reg["tstat"],
            title="Sign Regression — T-stat by Lag",
            save_path=out_dir / "tstat_sign.png",
            dpi=dpi,
        )
        logger.info("Regression charts saved.")

    # ── Single-stock mode ────────────────────────
    if args.symbol:
        sym = args.symbol.upper()
        logger.info("Running single-stock backtest: %s", sym)
        result = backtest_single(
            daily_prices[sym].dropna(),
            vol_target=vol_target, lookback=lookback,
            commission=0, margin_cap=margin_cap, ewm_com=ewm_com,
        )
        plot_single_backtest(result, sym, save_path=out_dir / f"backtest_{sym}.png", dpi=dpi)

        # Volatility/price chart
        daily_vol = daily_prices[sym].pct_change().dropna()
        from tsmom.volatility import exante_volatility
        vol_s = exante_volatility(daily_vol, com=ewm_com)
        plot_volatility_price(vol_s, daily_prices[sym], sym,
                              save_path=out_dir / f"vol_price_{sym}.png", dpi=dpi)

        print(f"\n{'='*40}")
        print(f"  {sym} — Buy & Hold")
        print(f"{'='*40}")
        print(calculate_metrics(result["raw_rets"]))
        print(f"\n{'='*40}")
        print(f"  {sym} — TSMOM Strategy")
        print(f"{'='*40}")
        print(calculate_metrics(result["strat_rets"]))
        return

    # ── Universe backtest ────────────────────────
    logger.info("Running universe backtest (%d stocks)…", len(daily_prices.columns))
    all_returns, all_metrics = backtest_universe(
        daily_prices,
        vol_target=vol_target, lookback=lookback,
        commission=commission, margin_cap=margin_cap, ewm_com=ewm_com,
    )

    # ── Portfolio charts ─────────────────────────
    plot_portfolio_cumret(
        all_returns,
        save_path=out_dir / "portfolio_cumret.png", dpi=dpi,
    )
    logger.info("Portfolio cumulative return chart saved.")

    # Distribution charts
    if "Annualized Return" in all_metrics.index:
        plot_distribution(
            all_metrics.loc["Annualized Return"].tolist(),
            "Distribution of Annualized Returns", "Annualized Return",
            save_path=out_dir / "dist_returns.png", dpi=dpi,
        )
        plot_distribution(
            all_metrics.loc["Annualized Volatility"].tolist(),
            "Distribution of Annualized Volatility", "Annualized Volatility",
            save_path=out_dir / "dist_volatility.png", dpi=dpi,
        )

    # ── Benchmark comparison ─────────────────────
    bench_path = cfg["data"].get("benchmark_csv", "data/vni.csv")
    bench = load_benchmark_csv(bench_path)
    if not bench.empty and "close" in bench.columns:
        bench_weekly = bench.resample("W").last().ffill()
        bench_weekly["vni"] = bench_weekly["close"].pct_change()
        tsmom_weekly = all_returns.mean(axis=1)
        plot_tsmom_vs_benchmark(
            tsmom_weekly, bench_weekly["vni"],
            save_path=out_dir / "tsmom_vs_vni.png", dpi=dpi,
        )
        logger.info("Benchmark comparison chart saved.")

    # ── Save metrics ─────────────────────────────
    metrics_path = out_dir / "metrics.csv"
    all_metrics.to_csv(metrics_path)
    logger.info("Metrics saved to %s", metrics_path)

    # ── Summary ──────────────────────────────────
    port_ret = all_returns.mean(axis=1)
    print(f"\n{'='*50}")
    print("  TSMOM Portfolio Summary (equal-weight)")
    print(f"{'='*50}")
    print(calculate_metrics(port_ret))
    print(f"\nCharts saved to: {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
