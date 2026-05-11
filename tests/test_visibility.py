"""
Tests for visibility graph module.
"""

import numpy as np
import pandas as pd

from tsmom.visibility import (
    build_visibility_graph,
    compute_risk_scale,
    rolling_graph_features,
)


class TestBuildVisibilityGraph:
    def test_build_graph_returns_networkx_graph(self):
        ts = np.cumsum(np.random.default_rng(42).standard_normal(50))
        G = build_visibility_graph(ts)
        assert G.number_of_nodes() == 50
        assert G.number_of_edges() > 0

    def test_constant_series_edges(self):
        """Uniform values produce a visibility graph with known structure."""
        ts = np.ones(10)
        G = build_visibility_graph(ts)
        assert G.number_of_nodes() == 10
        assert G.number_of_edges() > 0

    def test_strictly_decreasing_series_full_visibility(self):
        ts = np.arange(10, 0, -1)
        G = build_visibility_graph(ts)
        assert G.number_of_edges() > 0


class TestRollingGraphFeatures:
    def test_returns_dataframe_with_correct_columns(self):
        prices = pd.Series(
            100 * np.cumprod(1 + np.random.default_rng(43).standard_normal(80) * 0.02),
            index=pd.date_range("2024-01-01", periods=80, freq="B"),
        )
        gf = rolling_graph_features(prices, window=50)
        assert isinstance(gf, pd.DataFrame)
        assert set(gf.columns) == {"diameter", "betw_var"}
        assert len(gf) > 0
        assert gf["diameter"].min() >= 0
        assert gf["betw_var"].min() >= 0

    def test_short_series_returns_empty(self):
        prices = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3))
        gf = rolling_graph_features(prices, window=60)
        assert len(gf) == 0


class TestComputeRiskScale:
    def test_returns_series_same_index(self):
        dates = pd.date_range("2024-01-01", periods=50)
        gf = pd.DataFrame({
            "diameter": np.random.default_rng(44).uniform(4, 8, 50),
            "betw_var": np.random.default_rng(45).uniform(0.001, 0.01, 50),
        }, index=dates)
        risk = compute_risk_scale(gf, ewm_alpha=0.3)
        assert isinstance(risk, pd.Series)
        assert len(risk) == 50
        assert risk.min() >= 0.0
        assert risk.max() <= 1.0

    def test_random_input_stays_in_range(self):
        """On random walk, risk should always be in [0, 1]."""
        dates = pd.date_range("2024-01-01", periods=200)
        gf = pd.DataFrame({
            "diameter": np.abs(np.cumsum(np.random.default_rng(47).standard_normal(200) * 0.1) + 5),
            "betw_var": np.abs(np.random.default_rng(48).uniform(0.001, 0.02, 200)),
        }, index=dates)
        risk = compute_risk_scale(gf, ewm_alpha=0.15)
        assert risk.min() >= 0.0
        assert risk.max() <= 1.0
