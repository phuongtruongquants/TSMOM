"""
Tests for volatility estimation.
"""

import numpy as np
import pandas as pd

from tsmom.volatility import compute_weekly_volatility, exante_volatility


class TestExAnteVolatility:
    """Tests for the ex-ante volatility estimator."""

    def test_constant_returns_zero_volatility(self):
        """Volatility of constant returns should approach zero."""
        r = pd.Series(0.001, index=pd.date_range("2020-01-01", periods=200, freq="B"))
        vol = exante_volatility(r, com=60)
        # After enough data, vol of constant stream should be near zero
        assert vol.iloc[-1] < 0.01

    def test_known_variance(self):
        """Volatility of alternating ±x should have a known approximate value."""
        np.random.seed(42)
        r = pd.Series(
            np.random.randn(500) * 0.01,
            index=pd.date_range("2020-01-01", periods=500, freq="B"),
        )
        vol = exante_volatility(r, com=60)
        assert vol.notna().any()
        # Should be roughly in the ballpark of 0.01 * sqrt(252) ~ 0.16
        assert 0.05 < vol.iloc[-1] < 0.30

    def test_min_periods_enforced(self):
        """Should produce NaN until enough periods for vol estimation."""
        r = pd.Series(
            np.random.randn(300) * 0.01,
            index=pd.date_range("2020-01-01", periods=300, freq="B"),
        )
        vol = exante_volatility(r, com=60)
        # First 59 values should be NaN (min_periods=com)
        assert vol.iloc[:59].isna().all()
        assert vol.iloc[60:].notna().all()

    def test_dataframe_input(self):
        """exante_volatility should work on DataFrames."""
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        np.random.seed(42)
        df = pd.DataFrame(
            {"A": np.random.randn(300) * 0.01, "B": np.random.randn(300) * 0.02},
            index=dates,
        )
        vol = exante_volatility(df, com=60)
        assert isinstance(vol, pd.DataFrame)
        assert list(vol.columns) == ["A", "B"]
        assert vol.iloc[-1].notna().all()


class TestComputeWeeklyVolatility:
    """Tests for weekly volatility computation helper."""

    def test_weekly_from_daily_prices(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        np.random.seed(42)
        price = 100 * np.cumprod(1 + np.random.randn(200) * 0.01)
        prices = pd.DataFrame({"TEST": price}, index=dates)

        weekly_vol = compute_weekly_volatility(prices, com=60)
        assert isinstance(weekly_vol, pd.DataFrame)
        assert weekly_vol.index.inferred_type == "datetime64"
        # Weekly index should have fewer entries than daily
        assert len(weekly_vol) < len(prices)
