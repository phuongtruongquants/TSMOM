"""
Data loading utilities.

Supports two data sources:
  1. Local CSV file (default) — expects columns: timestamp, symbol, close
  2. vnstock / vnstock_data API — fetches directly from Vietnamese market data providers
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Default symbol list (60 stocks from the original notebook)
# ──────────────────────────────────────────────
DEFAULT_SYMBOLS = [
    "ACB", "ASM", "BVH", "CII", "CTD", "CTG", "DBC", "DIG", "DLG", "DPM",
    "DXG", "EIB", "FPT", "GAS", "GMD", "HAG", "HBC", "HDG", "HHS", "HNG",
    "HPG", "HSG", "HT1", "ITA", "KBC", "KDC", "KDH", "MBB", "MSN", "MWG",
    "NLG", "NT2", "NVL", "OGC", "PAN", "PDR", "PHR", "PNJ", "PPC", "PVD",
    "PVT", "REE", "SBT", "SSI", "STB", "TCM", "TSC", "TTF", "VCB", "VCG",
    "VHC", "VIC", "VIX", "VND", "VNM", "SJS", "HAH", "GEX", "DCM", "DGW",
]


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load stock price data from a local CSV.

    Expected CSV format:
        timestamp,symbol,close
        2014-02-06,ACB,2.83
        ...

    OR a pivoted format with columns = symbols and index = dates.

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame: index=DatetimeIndex('timestamp'), columns=symbols, values=close
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path.resolve()}\n"
            "Run `python scripts/fetch_data.py` first, or place your CSV at this path."
        )

    df = pd.read_csv(path)

    # Detect format: if 'symbol' column exists → long format, needs pivot
    if "symbol" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.pivot_table(index="timestamp", columns="symbol", values="close")
    else:
        # Assume first column is date index
        df.index = pd.to_datetime(df.iloc[:, 0] if df.index.dtype == "int64" else df.index)
        if df.columns[0].lower() in ("timestamp", "date", "time"):
            df = df.set_index(df.columns[0])
            df.index = pd.to_datetime(df.index)

    df = df.ffill()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    logger.info("Loaded %d rows × %d symbols from %s", len(df), len(df.columns), csv_path)
    return df


def load_from_vnstock(
    symbols: list[str] | None = None,
    start: str = "2014-01-01",
    end: str = "2025-03-20",
) -> pd.DataFrame:
    """
    Fetch daily close prices from vnstock / vnstock_data.

    Parameters
    ----------
    symbols : list[str], optional
        Ticker symbols to download.  Defaults to DEFAULT_SYMBOLS.
    start, end : str
        Date range in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame: index=DatetimeIndex, columns=symbols, values=close
    """
    # Try vnstock_data first, fall back to vnstock
    try:
        from vnstock_data import Vnstock
        logger.info("Using vnstock_data backend")
    except ImportError:
        try:
            from vnstock import Vnstock
            logger.info("Using vnstock backend")
        except ImportError:
            raise ImportError(
                "Neither vnstock_data nor vnstock is installed.\n"
                "Install with:  pip install vnstock   or   pip install vnstock_data"
            )

    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    frames = {}
    for sym in symbols:
        try:
            stock = Vnstock().stock(symbol=sym, source="VCI")
            hist = stock.quote.history(start=start, end=end, interval="1D")
            if hist is not None and not hist.empty:
                # Normalize column name
                close_col = [c for c in hist.columns if c.lower() == "close"]
                time_col = [c for c in hist.columns if c.lower() in ("time", "timestamp", "date")]
                if close_col and time_col:
                    series = hist.set_index(time_col[0])[close_col[0]]
                    series.index = pd.to_datetime(series.index)
                    frames[sym] = series
                    logger.info("  ✓ %s: %d rows", sym, len(series))
        except Exception as exc:
            logger.warning("  ✗ %s: %s", sym, exc)

    if not frames:
        raise RuntimeError("No data fetched. Check network and vnstock installation.")

    df = pd.DataFrame(frames).ffill()
    df.index.name = "timestamp"
    logger.info("Fetched %d rows × %d symbols via vnstock", len(df), len(df.columns))
    return df


def load_benchmark_csv(csv_path: str) -> pd.DataFrame:
    """
    Load VN-Index benchmark data from CSV.

    Expects columns including 'close' and a date index or 'time'/'timestamp' column.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Benchmark file not found: %s — skipping benchmark analysis", path)
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Try to identify timestamp column
    for col in ("time", "timestamp", "date"):
        if col in df.columns:
            df = df.set_index(col)
            break

    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    return df


def load_data(config: dict) -> pd.DataFrame:
    """
    Load data based on config['data']['source'].

    Parameters
    ----------
    config : dict
        Parsed YAML config.

    Returns
    -------
    pd.DataFrame
        Daily close prices, pivoted: index=dates, columns=symbols.
    """
    source = config["data"].get("source", "csv")
    if source == "csv":
        return load_from_csv(config["data"]["csv_path"])
    elif source == "vnstock":
        return load_from_vnstock(
            start=config["data"].get("start_date", "2014-01-01"),
            end=config["data"].get("end_date", "2025-03-20"),
        )
    else:
        raise ValueError(f"Unknown data source: {source!r}. Use 'csv' or 'vnstock'.")
