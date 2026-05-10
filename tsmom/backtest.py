"""
Backtesting engine for the TSMOM strategy.

For each stock:
  1. Compute daily → weekly returns and ex-ante volatility
  2. Generate momentum signal
  3. Size positions to target a fixed annualized volatility
  4. Calculate strategy returns net of transaction costs
"""

import logging

import pandas as pd

from .metrics import calculate_metrics
from .signal import compute_signal
from .volatility import exante_volatility

logger = logging.getLogger(__name__)


def backtest_single(
    daily_price: pd.Series,
    vol_target: float = 0.4,
    lookback: int = 6,
    commission: float = 0.001,
    margin_cap: float = 2.0,
    ewm_com: int = 60,
) -> pd.DataFrame:
    """
    Backtest the TSMOM strategy on a single stock.

    Parameters
    ----------
    daily_price : pd.Series
        Daily close prices for one stock.
    vol_target : float
        Annualized volatility target (default 0.40).
    lookback : int
        Weeks of lookback for momentum signal (default 6).
    commission : float
        One-way transaction cost (default 0.001 = 0.1%).
    margin_cap : float
        Maximum leverage allowed (default 2.0).
    ewm_com : int
        Center of mass for volatility estimation (default 60).

    Returns
    -------
    pd.DataFrame
        Columns: raw_rets, signal, position_size, position, trade_cost, strat_rets
    """
    daily_return = daily_price.pct_change().dropna()
    weekly_return = daily_price.resample("W").last().pct_change().dropna()

    daily_vol = exante_volatility(daily_return, com=ewm_com)
    weekly_vol = daily_vol.resample("W").last().ffill()

    # Position sizing: scale to target vol, cap at margin_cap
    position_size = (vol_target / weekly_vol).clip(upper=margin_cap)

    signal = compute_signal(weekly_return, window=lookback)
    position = signal * position_size

    # Round-trip cost on position changes, charged when the new position is actually held
    trade_cost = 2 * commission * position.diff().abs().shift(1)

    # Strategy return: prior week's position × this week's return − cost
    strat_ret = position.shift(1) * weekly_return - trade_cost

    result = pd.DataFrame({
        "raw_rets": weekly_return,
        "signal": signal,
        "position_size": position_size,
        "position": position,
        "trade_cost": trade_cost,
        "strat_rets": strat_ret,
    })

    return result.dropna()


def summarize_universe_returns(all_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Aggregate per-symbol returns into equal-weight portfolio returns and metrics."""
    portfolio_returns = all_returns.mean(axis=1).dropna()
    portfolio_metrics = calculate_metrics(portfolio_returns)
    return portfolio_returns, portfolio_metrics


def backtest_universe(
    daily_prices: pd.DataFrame,
    vol_target: float = 0.4,
    lookback: int = 6,
    commission: float = 0.001,
    margin_cap: float = 2.0,
    ewm_com: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the TSMOM backtest across all stocks in the universe.

    Parameters
    ----------
    daily_prices : pd.DataFrame
        Daily close prices: index=dates, columns=symbols.
    vol_target, lookback, commission, margin_cap, ewm_com :
        Strategy parameters (see `backtest_single`).

    Returns
    -------
    all_returns : pd.DataFrame
        Strategy returns per symbol: index=weekly dates, columns=symbols.
    all_metrics : pd.DataFrame
        Performance metrics per symbol.
    """
    all_returns = pd.DataFrame()
    all_metrics = pd.DataFrame()

    for symbol in daily_prices.columns:
        try:
            result = backtest_single(
                daily_prices[symbol].dropna(),
                vol_target=vol_target,
                lookback=lookback,
                commission=commission,
                margin_cap=margin_cap,
                ewm_com=ewm_com,
            )
            all_returns[symbol] = result["strat_rets"]

            metrics = calculate_metrics(result["strat_rets"])
            all_metrics[symbol] = metrics["metrics"]
        except (ValueError, KeyError, IndexError) as exc:
            logger.warning("Skipping %s: %s: %s", symbol, type(exc).__name__, exc)

    return all_returns, all_metrics
