"""
Momentum signal generation.

Computes a long-only trading signal based on rolling cumulative returns:

    signal_t = max(0, sign(Π_{i=t-W}^{t} (1 + r_i) - 1))

where W is the lookback window (default 6 weeks).
"""

import numpy as np
import pandas as pd


def compute_signal(
    returns: pd.Series | pd.DataFrame,
    window: int = 6,
) -> pd.Series | pd.DataFrame:
    """
    Compute the momentum trading signal.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period (typically weekly) returns.
    window : int
        Number of periods for the rolling cumulative return (default 6).

    Returns
    -------
    pd.Series or pd.DataFrame
        Signal values: +1 (long) or 0 (flat).
        Negative signals are clipped to 0 (long-only strategy).
    """
    cum_rets = returns.rolling(window=window).apply(
        lambda x: np.prod(1 + x) - 1, raw=True
    )
    signal = np.sign(cum_rets).clip(lower=0)
    return signal
