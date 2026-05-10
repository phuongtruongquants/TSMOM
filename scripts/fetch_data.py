#!/usr/bin/env python3
"""
Fetch stock price data and save to local CSV for reproducible backtesting.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --symbols ACB HPG VNM
    python scripts/fetch_data.py --start 2015-01-01 --end 2025-01-01
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsmom.data import load_from_vnstock, DEFAULT_SYMBOLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch VN stock data via vnstock")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to fetch")
    parser.add_argument("--start", default="2014-01-01", help="Start date")
    parser.add_argument("--end", default="2026-05-11", help="End date")
    parser.add_argument("--output", default="data/stock_prices.csv", help="Output CSV path")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS
    logger.info("Fetching %d symbols from %s to %s", len(symbols), args.start, args.end)

    df = load_from_vnstock(symbols=symbols, start=args.start, end=args.end)

    # Save in long format: timestamp, symbol, close
    long_df = df.reset_index().melt(id_vars="timestamp", var_name="symbol", value_name="close")
    long_df = long_df.dropna(subset=["close"]).sort_values(["symbol", "timestamp"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, index=False)
    logger.info("Saved %d rows × %d columns to %s", len(long_df), len(long_df.columns), out_path)


if __name__ == "__main__":
    main()
