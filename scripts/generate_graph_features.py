#!/usr/bin/env python3
"""
Pre-compute visibility-graph features for risk management.

Outputs:
    data/graph_features.csv  — daily diameter + betweenness variance
    data/risk_scale.csv      — weekly risk scale aligned with backtest index

Usage:
    python scripts/generate_graph_features.py
    python scripts/generate_graph_features.py --symbol HPG
    python scripts/generate_graph_features.py --window 90 --symbol VNINDEX
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsmom.visibility import compute_risk_scale, rolling_graph_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate visibility-graph features")
    parser.add_argument("--symbol", default="VNINDEX",
                        help="Symbol or 'VNINDEX' for benchmark")
    parser.add_argument("--window", type=int, default=60,
                        help="Rolling window size (days)")
    parser.add_argument("--output-graph", default="data/graph_features.csv",
                        help="Output path for daily graph features")
    parser.add_argument("--output-risk", default="data/risk_scale.csv",
                        help="Output path for weekly risk scale")
    args = parser.parse_args()

    # Load price data
    if args.symbol == "VNINDEX":
        vni_path = Path("data/vni.csv")
        if not vni_path.exists():
            logger.error("data/vni.csv not found. Fetch VN-Index data first.")
            sys.exit(1)
        prices = pd.read_csv(vni_path, index_col=0, parse_dates=True)
        prices = prices["close"]
        logger.info("Loaded VN-Index: %d rows", len(prices))
    else:
        csv_path = Path("data/stock_prices.csv")
        if not csv_path.exists():
            logger.error("data/stock_prices.csv not found.")
            sys.exit(1)
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        stock = df[df["symbol"] == args.symbol].set_index("timestamp")
        prices = stock["close"]
        logger.info("Loaded %s: %d rows", args.symbol, len(prices))

    # Compute rolling graph features
    logger.info("Computing rolling graph features (window=%d)...", args.window)
    gf = rolling_graph_features(prices, window=args.window)
    logger.info("Graph features: %d rows", len(gf))

    # Save daily features
    Path(args.output_graph).parent.mkdir(parents=True, exist_ok=True)
    gf.to_csv(args.output_graph)
    logger.info("Saved daily graph features to %s", args.output_graph)

    # Compute weekly risk scale
    risk = compute_risk_scale(gf)
    risk_weekly = risk.resample("W").last().ffill()
    risk_weekly.name = "risk_scale"

    Path(args.output_risk).parent.mkdir(parents=True, exist_ok=True)
    risk_weekly.to_csv(args.output_risk, header=True)
    logger.info("Saved weekly risk scale to %s (%d values)", args.output_risk, len(risk_weekly))
    logger.info("Risk scale summary: mean=%.3f  min=%.3f  pct_below_1=%.1f%%",
                risk_weekly.mean(), risk_weekly.min(),
                (risk_weekly < 1.0).mean() * 100)


if __name__ == "__main__":
    main()
