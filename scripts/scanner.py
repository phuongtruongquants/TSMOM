#!/usr/bin/env python3
"""
Stock Scanner — classify every stock into regime + risk level.

Usage:
    python scripts/scanner.py                    # scan all 60 stocks
    python scripts/scanner.py --symbols HPG VNM  # scan specific stocks
    python scripts/scanner.py --lookback 90      # use 90-day window
    python scripts/scanner.py --output stdout    # print table instead of CSV
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsmom.visibility import scan_universe
from tsmom.data import load_data, DEFAULT_SYMBOLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Scan stocks for regime and risk")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to scan")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window (days)")
    parser.add_argument("--output", default="stdout", help="Output: 'stdout' or 'csv' path")
    args = parser.parse_args()

    # Load price data
    config = {
        "data": {"source": "csv", "csv_path": "data/stock_prices.csv"}
    }
    prices = load_data(config)

    if args.symbols:
        prices = prices[[s for s in args.symbols if s in prices.columns]]
        if prices.empty:
            logger.error("No matching symbols found.")
            sys.exit(1)

    logger.info("Scanning %d stocks (lookback=%d days)...", len(prices.columns), args.lookback)
    report = scan_universe(prices, lookback_days=args.lookback)

    if report.empty:
        logger.warning("No results.")
        sys.exit(1)

    # Summary
    n_danger = (report["risk_level"] == "danger").sum()
    n_warning = (report["risk_level"] == "warning").sum()
    n_normal = (report["risk_level"] == "normal").sum()
    logger.info("Results: %d normal | %d warning | %d danger", n_normal, n_warning, n_danger)

    if args.output == "stdout":
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.width", 160)
        pd.set_option("display.max_columns", 12)
        print(report.to_string(index=False))
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)
        logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
