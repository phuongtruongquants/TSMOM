"""
Momentum regression tests.

Provides pooled OLS regressions to test whether past returns predict future
vol-adjusted returns — the core statistical evidence for time-series momentum.

Two regression specifications:

1. **TSMOM regression** (continuous):
   (r_{t+h} / σ_{t+h-1}) = α + β_h · (r_t / σ_{t-1}) + ε

2. **Sign regression** (binary signal):
   (r_t / σ_{t-1}) = α + β_h · sign(r_{t-h}) + ε
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_tsmom_regressions(
    returns: pd.DataFrame,
    vol: pd.DataFrame,
    max_lag: int = 12,
) -> pd.DataFrame:
    """
    Pooled OLS of future vol-adjusted returns on current vol-adjusted returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Weekly returns: index=dates, columns=symbols.
    vol : pd.DataFrame
        Weekly volatility, aligned with `returns`.
    max_lag : int
        Number of lags to test.

    Returns
    -------
    pd.DataFrame
        Columns: ['beta', 'tstat'], indexed by lag (1 … max_lag).
    """
    betas = []
    tstats = []

    # Standardized returns: Y_t = r_t / σ_{t-1}
    Y = returns / vol.shift(1)

    for lag in range(1, max_lag + 1):
        Y_future = Y.shift(-lag)
        X_current = Y.copy()

        # Pool across stocks & time
        df = pd.DataFrame({
            "Y_future": Y_future.stack(),
            "X_current": X_current.stack(),
        }).dropna()

        if df.empty:
            break

        df = sm.add_constant(df)
        model = sm.OLS(df["Y_future"], df[["const", "X_current"]])
        results = model.fit()

        betas.append(results.params["X_current"])
        tstats.append(results.tvalues["X_current"])

    return pd.DataFrame(
        {"beta": betas, "tstat": tstats},
        index=range(1, len(betas) + 1),
    )


def run_sign_regressions(
    returns: pd.DataFrame,
    vol: pd.DataFrame,
    max_lag: int = 12,
) -> pd.DataFrame:
    """
    Pooled OLS of vol-adjusted returns on the *sign* of past returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Weekly returns.
    vol : pd.DataFrame
        Weekly volatility.
    max_lag : int
        Number of lags to test.

    Returns
    -------
    pd.DataFrame
        Columns: ['beta', 'tstat'], indexed by lag (1 … max_lag).
    """
    betas = []
    tstats = []

    Y = returns / vol.shift(1)

    for h in range(1, max_lag + 1):
        X_sign = np.sign(returns.shift(h))

        df = pd.DataFrame({
            "Y": Y.stack(),
            "X_sign": X_sign.stack(),
        }).dropna()

        if df.empty:
            break

        df = sm.add_constant(df)
        model = sm.OLS(df["Y"], df[["const", "X_sign"]])
        results = model.fit()

        betas.append(results.params["X_sign"])
        tstats.append(results.tvalues["X_sign"])

    return pd.DataFrame(
        {"beta": betas, "tstat": tstats},
        index=range(1, len(betas) + 1),
    )
