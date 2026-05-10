"""
Performance metrics calculation.
"""

import numpy as np
import pandas as pd

METRIC_ORDER = [
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio",
    "Max Drawdown",
    "Skewness",
    "Kurtosis",
    "Positive Weeks",
]


def summarize_metrics(returns, periods_per_year=52):
    """Calculate annualized return, vol, Sharpe, max DD, skew, kurtosis."""
    clean_returns = pd.Series(returns).dropna()
    if clean_returns.empty:
        return {metric: np.nan for metric in METRIC_ORDER}

    ann_ret = clean_returns.mean() * periods_per_year
    ann_vol = clean_returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0.0
    cum = (1 + clean_returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    return {
        "Annualized Return": round(ann_ret, 4),
        "Annualized Volatility": round(ann_vol, 4),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown": round(max_dd, 4),
        "Skewness": round(clean_returns.skew(), 2),
        "Kurtosis": round(clean_returns.kurtosis(), 2),
        "Positive Weeks": round((clean_returns > 0).mean(), 4),
    }


def calculate_metrics(returns, periods_per_year=52):
    summary = summarize_metrics(returns, periods_per_year=periods_per_year)
    return pd.DataFrame(summary, index=["metrics"]).T
