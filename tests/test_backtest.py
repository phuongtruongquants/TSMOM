"""
Tests for the backtesting engine.
"""

import numpy as np
import pandas as pd

from tsmom.backtest import backtest_single, backtest_universe, summarize_universe_returns


def _make_price(start=100, n=500, seed=42):
    """Generate a synthetic price series."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = start * np.cumprod(1 + np.random.randn(n) * 0.015)
    return pd.Series(price, index=dates)


class TestBacktestSingle:
    """Tests for single-stock backtest."""

    def test_returns_dataframe_with_expected_columns(self):
        result = backtest_single(_make_price())
        expected_cols = {"raw_rets", "signal", "position_size", "position",
                         "trade_cost", "strat_rets"}
        assert expected_cols.issubset(result.columns)

    def test_no_future_peeking(self):
        """Changing future prices must not change earlier positions."""
        price = _make_price(n=700)
        split_idx = price.index[450]

        altered_price = price.copy()
        altered_price.loc[altered_price.index > split_idx] *= 1.5

        result = backtest_single(price, lookback=6)
        altered_result = backtest_single(altered_price, lookback=6)

        shared_index = result.index.intersection(altered_result.index)
        earlier_index = shared_index[shared_index <= split_idx]

        assert len(earlier_index) > 20
        pd.testing.assert_series_equal(
            result.loc[earlier_index, "position"],
            altered_result.loc[earlier_index, "position"],
            check_names=False,
        )

    def test_result_length_reasonable(self):
        """Backtest should produce a non-trivial number of data points."""
        result = backtest_single(_make_price(n=600))
        assert len(result) > 20

    def test_commission_reduces_returns(self):
        """Higher commission should mean lower net returns."""
        result_no_cost = backtest_single(_make_price(), commission=0.0)
        result_cost = backtest_single(_make_price(), commission=0.01)

        cum_0 = (1 + result_no_cost["strat_rets"].dropna()).prod()
        cum_c = (1 + result_cost["strat_rets"].dropna()).prod()
        assert cum_0 >= cum_c


class TestBacktestUniverse:
    """Tests for universe-wide backtest."""

    def test_multiple_symbols(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        prices = {}
        for sym, seed_val in [("AAA", 1), ("BBB", 2), ("CCC", 3)]:
            np.random.seed(seed_val)
            prices[sym] = 100 * np.cumprod(1 + np.random.randn(500) * 0.015)
        df = pd.DataFrame(prices, index=dates)

        all_returns, all_metrics = backtest_universe(df)
        assert len(all_returns.columns) == 3
        assert not all_returns.empty
        assert not all_metrics.empty

    def test_handles_symbol_with_short_history(self):
        """Symbols with too little data should be skipped gracefully."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        df = pd.DataFrame(index=dates)
        df["GOOD"] = 100 * np.cumprod(1 + np.random.randn(500) * 0.015)
        # "SHORT" has only 20 days — not enough for any meaningful backtest
        short_price = 50 * np.cumprod(1 + np.random.randn(20) * 0.01)
        df["SHORT"] = pd.Series(short_price, index=dates[:20])

        all_returns, all_metrics = backtest_universe(df.ffill())
        # GOOD should be in the results; SHORT may or may not depending on ffill
        assert "GOOD" in all_returns.columns
        assert len(all_returns) > 0

    def test_summarize_universe_returns_returns_portfolio_series_and_metrics(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        prices = pd.DataFrame({
            "AAA": 100 * np.cumprod(1 + np.random.randn(500) * 0.015),
            "BBB": 110 * np.cumprod(1 + np.random.randn(500) * 0.012),
        }, index=dates)

        all_returns, _ = backtest_universe(prices)
        portfolio_returns, portfolio_metrics = summarize_universe_returns(all_returns)

        assert not portfolio_returns.empty
        assert portfolio_returns.index.equals(all_returns.index)
        assert "Sharpe Ratio" in portfolio_metrics.index
