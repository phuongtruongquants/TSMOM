"""
Performance metrics calculation.
"""

import numpy as np
import pandas as pd


def calculate_metrics(returns, periods_per_year=52):
    """Calculate annualized return, vol, Sharpe, max DD, skew, kurtosis."""
    ann_ret = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0.0
    cum = (1 + returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    summary = pd.DataFrame({
        "Annualized Return": round(ann_ret, 4),
        "Annualized Volatility": round(ann_vol, 4),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown": round(max_dd, 4),
        "Skewness": round(returns.skew(), 2),
        "Kurtosis": round(returns.kurtosis(), 2),
    }, index=["metrics"])
    return summary.T
