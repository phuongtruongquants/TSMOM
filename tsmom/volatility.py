"""
Ex-ante volatility estimation using exponentially weighted variance.

Based on the framework in Moskowitz, Ooi & Pedersen (2012):

    σ²_t = annualization × EWM_var(r_{t-1}, com)

where the center-of-mass (com) controls the decay rate of the
exponential weights:  α = 1 / (com + 1).
"""

import numpy as np
import pandas as pd


def exante_volatility(
    returns: pd.Series | pd.DataFrame,
    com: int = 60,
    annualization: int = 252,
) -> pd.Series | pd.DataFrame:
    """
    Compute ex-ante annualized volatility.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily asset returns.
    com : int
        Center of mass for the exponential weights (default 60).
    annualization : int
        Number of trading days per year (default 252).

    Returns
    -------
    pd.Series or pd.DataFrame
        Annualized volatility σ_t, using returns up to t.
    """
    ewm_variance = returns.ewm(com=com, adjust=False, min_periods=com).var()
    sigma = np.sqrt(annualization * ewm_variance)
    return sigma


def compute_weekly_volatility(
    daily_prices: pd.DataFrame,
    com: int = 60,
    annualization: int = 252,
) -> pd.DataFrame:
    """
    Compute weekly ex-ante volatility from daily prices.

    Steps:
      1. Compute daily returns
      2. Compute daily ex-ante vol via EWM
      3. Resample to weekly (last obs)

    Parameters
    ----------
    daily_prices : pd.DataFrame
        Daily close prices: index=dates, columns=symbols.
    com, annualization : int
        Passed through to `exante_volatility`.

    Returns
    -------
    pd.DataFrame
        Weekly volatility estimates.
    """
    daily_returns = daily_prices.pct_change().dropna()
    daily_vol = daily_returns.apply(exante_volatility, com=com, annualization=annualization)
    weekly_vol = daily_vol.resample("W").last().ffill()
    return weekly_vol
