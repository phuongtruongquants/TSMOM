"""
Tests for momentum signal generation.
"""

import numpy as np
import pandas as pd

from tsmom.signal import compute_signal


class TestComputeSignal:
    """Tests for the momentum signal generator."""

    def test_positive_momentum_long_signal(self):
        """All-positive returns should produce a long signal."""
        r = pd.Series(0.01, index=range(20))
        signal = compute_signal(r, window=6)
        assert (signal.dropna() == 1).all()

    def test_negative_momentum_flat_signal(self):
        """All-negative returns should produce a flat (0) signal (long-only)."""
        r = pd.Series(-0.01, index=range(20))
        signal = compute_signal(r, window=6)
        assert (signal.dropna() == 0).all()

    def test_output_is_0_or_1(self):
        """Signal should only contain 0 or 1 for a long-only strategy."""
        np.random.seed(42)
        r = pd.Series(np.random.randn(100) * 0.02, index=range(100))
        signal = compute_signal(r, window=6)
        valid = signal.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_dataframe_input(self):
        """compute_signal should work on DataFrames."""
        np.random.seed(42)
        df = pd.DataFrame(
            {"A": np.random.randn(50) * 0.02, "B": np.random.randn(50) * 0.01},
            index=range(50),
        )
        signal = compute_signal(df, window=6)
        assert isinstance(signal, pd.DataFrame)
        assert list(signal.columns) == ["A", "B"]

    def test_less_data_than_window_produces_nan(self):
        """Signal should be NaN when there isn't enough data for the window."""
        r = pd.Series(np.random.randn(10) * 0.01, index=range(10))
        signal = compute_signal(r, window=6)
        assert signal.iloc[:5].isna().all()  # First 5 have < 6 obs
        assert signal.iloc[5:].notna().all()
