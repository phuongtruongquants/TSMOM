"""
Visibility Graph — regime detection and risk management for time series.

Converts price series into networks (natural visibility graph), then
extracts structural features (diameter, betweenness centrality variance)
as early-warning risk indicators.

Based on vnquant.vn approach: visibility graph + Louvain community detection
for regime identification, graph features for piecewise-linear risk scaling.

References:
    Lacasa et al. (2008) "From time series to complex networks: The
    visibility graph" — https://arxiv.org/abs/0810.0920
"""

import logging

import networkx as nx
import numpy as np
import pandas as pd
from ts2vg import NaturalVG

logger = logging.getLogger(__name__)


def build_visibility_graph(series: np.ndarray) -> nx.Graph:
    """Build a natural visibility graph from a 1-D time series."""
    vg = NaturalVG(directed=None)
    return vg.build(series).as_networkx()


def detect_regimes(prices: pd.Series) -> tuple[list[set[int]], list[str], list[int]]:
    graph = build_visibility_graph(prices.values)
    communities = nx.algorithms.community.louvain_communities(graph, seed=42)
    colors = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#937860", "#DA8BC3", "#CCB974", "#64B5CD", "#E24A33",
    ]
    node_colors = ["#000000"] * len(prices)
    for cid, nodes in enumerate(communities):
        color = colors[cid % len(colors)]
        for n in nodes:
            node_colors[n] = color
    return communities, node_colors, colors


def rolling_graph_features(
    prices: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Compute daily rolling visibility-graph features.

    For each window of the last `window` days:
      - diameter: longest shortest path in the visibility graph
      - betw_var: variance of betweenness centrality across nodes

    Parameters
    ----------
    prices : pd.Series
        Daily close prices with datetime index.
    window : int
        Number of days in rolling window (default 60).

    Returns
    -------
    pd.DataFrame
        Columns ['diameter', 'betw_var'], indexed by date (starting at window).
    """
    n = len(prices)
    if n < window:
        return pd.DataFrame(columns=["diameter", "betw_var"])

    values = prices.values.astype(float)
    dates = prices.index[window - 1 :]
    vg = NaturalVG(directed=None)

    result: dict[str, list[float]] = {"diameter": [], "betw_var": []}

    for i in range(window, n + 1):
        win = values[i - window : i]
        G = vg.build(win).as_networkx()

        # Diameter — handle disconnected graphs
        if nx.is_connected(G):
            d = nx.diameter(G)
        else:
            largest = max(nx.connected_components(G), key=len)
            d = nx.diameter(G.subgraph(largest))

        # Betweenness centrality variance
        bc = np.fromiter(nx.betweenness_centrality(G).values(), dtype=float)
        bc_var = float(bc.var())

        result["diameter"].append(float(d))
        result["betw_var"].append(bc_var)

    return pd.DataFrame(result, index=dates)


def compute_risk_scale(
    graph_features: pd.DataFrame,
    z_threshold: float = 1.645,
    z_max: float = 2.576,
    s_min: float = 0.2,
    ewm_alpha: float = 0.1,
) -> pd.Series:
    """Convert graph features into a [0, 1] risk-scaling multiplier.

    Steps:
      1. EWM-normalize diameter and betw_var → Z-scores
      2. z_spike = max(z_diam, z_betw), clipped at 0
      3. Piecewise-linear: 1.0 below z_threshold, ramp to s_min at z_max

    Parameters
    ----------
    graph_features : pd.DataFrame
        Columns ['diameter', 'betw_var'] with datetime index.
    z_threshold : float
        Z-score at which risk scaling begins (default 1.645 = 90th pct).
    z_max : float
        Z-score at which risk scaling reaches floor (default 2.576 = 99th pct).
    s_min : float
        Minimum scale factor when risk is extreme (default 0.2).
    ewm_alpha : float
        Smoothing parameter for the EWM statistics (default 0.1).

    Returns
    -------
    pd.Series
        Risk scale multipliers, same index as input. 1.0 = full position.
    """
    gf = graph_features.dropna()
    if gf.empty:
        return pd.Series(dtype=float)

    # EWM mean and std for normalization
    mu_diam = gf["diameter"].ewm(alpha=ewm_alpha, min_periods=10).mean()
    sd_diam = gf["diameter"].ewm(alpha=ewm_alpha, min_periods=10).std()
    mu_betw = gf["betw_var"].ewm(alpha=ewm_alpha, min_periods=10).mean()
    sd_betw = gf["betw_var"].ewm(alpha=ewm_alpha, min_periods=10).std()

    # Z-scores, clipped at 0 (only care about spikes, not dips)
    z_diam = ((gf["diameter"] - mu_diam) / sd_diam.replace(0, np.nan)).clip(lower=0)
    z_betw = ((gf["betw_var"] - mu_betw) / sd_betw.replace(0, np.nan)).clip(lower=0)
    z_spike = pd.concat([z_diam, z_betw], axis=1).max(axis=1)

    # Piecewise-linear ramp
    risk = pd.Series(1.0, index=gf.index)
    ramp_zone = (z_spike > z_threshold) & (z_spike <= z_max)
    extreme = z_spike > z_max

    ramp = ((z_spike[ramp_zone] - z_threshold) / (z_max - z_threshold))
    risk[ramp_zone] = 1.0 - ramp * (1.0 - s_min)
    risk[extreme] = s_min

    return risk
