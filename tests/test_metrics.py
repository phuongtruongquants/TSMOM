"""
Tests for performance metrics calculation.
"""

import numpy as np
import pandas as pd

from tsmom.metrics import calculate_metrics, summarize_metrics


class TestCalculateMetrics:
    """Tests for calculate_metrics."""

    def test_positive_returns_positive_sharpe(self):
        """A consistently positive return stream should have positive Sharpe."""
        r = pd.Series(0.005, index=range(100))
        result = calculate_metrics(r)
        assert result.loc["Sharpe Ratio", "metrics"] > 0

    def test_zero_returns_zero_sharpe(self):
        """A zero-return stream should have zero Sharpe (not NaN)."""
        r = pd.Series(0.0, index=range(100))
        result = calculate_metrics(r)
        assert result.loc["Sharpe Ratio", "metrics"] == 0.0

    def test_negative_returns_negative_sharpe(self):
        """A consistently negative return stream should have negative Sharpe."""
        r = pd.Series(-0.005, index=range(100))
        result = calculate_metrics(r)
        assert result.loc["Sharpe Ratio", "metrics"] < 0.0

    def test_max_drawdown_non_positive(self):
        """Max drawdown should be <= 0."""
        np.random.seed(42)
        r = pd.Series(np.random.randn(200) * 0.01, index=range(200))
        result = calculate_metrics(r)
        assert result.loc["Max Drawdown", "metrics"] <= 0.0

    def test_returns_all_expected_metrics(self):
        r = pd.Series(np.random.randn(100) * 0.01, index=range(100))
        result = calculate_metrics(r)
        expected = ["Annualized Return", "Annualized Volatility", "Sharpe Ratio",
                    "Max Drawdown", "Skewness", "Kurtosis", "Positive Weeks"]
        for metric in expected:
            assert metric in result.index

    def test_summarize_metrics_returns_positive_weeks(self):
        r = pd.Series([0.02, -0.01, 0.03, 0.00])
        result = summarize_metrics(r)
        assert result["Positive Weeks"] == 0.5
