"""
Visibility Graph — regime detection and risk management for time series.

Converts price series into networks (natural visibility graph), then
extracts structural features as regime indicators and risk signals.

Two use-cases:
  1. **Scanner** — classify every stock into regime + risk level daily
  2. **Risk Management** — graph features → piecewise-linear risk scaling
     integrated into TSMOM backtest.

Based on vnquant.vn (2025-06-23): visibility graph + Louvain community
detection for regime, graph features for early-warning risk filter.

References:
    Lacasa et al. (2008) "From time series to complex networks"
    https://arxiv.org/abs/0810.0920
"""

import logging

import networkx as nx
import numpy as np
import pandas as pd
from ts2vg import NaturalVG

logger = logging.getLogger(__name__)

REGIME_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#CCB974", "#64B5CD", "#E24A33",
]

# ── Core visibility-graph tools ──────────────────────────


def build_visibility_graph(series: np.ndarray) -> nx.Graph:
    """Build a natural visibility graph from a 1-D time series."""
    vg = NaturalVG(directed=None)
    return vg.build(np.asarray(series).copy()).as_networkx()


def detect_regimes(prices: pd.Series) -> tuple[list[set[int]], list[str]]:
    """Louvain community detection on visibility graph → regime labels."""
    graph = build_visibility_graph(prices.values)
    communities = nx.algorithms.community.louvain_communities(graph, seed=42)

    node_colors = ["#000000"] * len(prices)
    for cid, nodes in enumerate(communities):
        color = REGIME_COLORS[cid % len(REGIME_COLORS)]
        for n in nodes:
            node_colors[n] = color

    return communities, node_colors


def _graph_metrics(graph: nx.Graph) -> dict[str, float]:
    """Extract diameter and betweenness variance from a single graph."""
    if nx.is_connected(graph):
        d = nx.diameter(graph)
    else:
        largest = max(nx.connected_components(graph), key=len)
        d = nx.diameter(graph.subgraph(largest))

    bc = np.fromiter(nx.betweenness_centrality(graph).values(), dtype=float)
    return {"diameter": float(d), "betw_var": float(bc.var())}


def rolling_graph_features(prices: pd.Series, window: int = 60) -> pd.DataFrame:
    """Daily rolling visibility-graph features (diameter, betw_var)."""
    n = len(prices)
    if n < window:
        return pd.DataFrame(columns=["diameter", "betw_var"])

    values = prices.values.astype(float)
    dates = prices.index[window - 1 :]
    vg = NaturalVG(directed=None)
    result: dict[str, list[float]] = {"diameter": [], "betw_var": []}

    for i in range(window, n + 1):
        win = values[i - window : i]
        m = _graph_metrics(vg.build(win).as_networkx())
        result["diameter"].append(m["diameter"])
        result["betw_var"].append(m["betw_var"])

    return pd.DataFrame(result, index=dates)


def compute_risk_scale(
    graph_features: pd.DataFrame,
    z_threshold: float = 1.645,
    z_max: float = 2.576,
    s_min: float = 0.2,
    ewm_alpha: float = 0.1,
) -> pd.Series:
    """Convert graph features into [0, 1] risk-scaling multiplier."""
    gf = graph_features.dropna()
    if gf.empty:
        return pd.Series(dtype=float)

    mu_diam = gf["diameter"].ewm(alpha=ewm_alpha, min_periods=10).mean()
    sd_diam = gf["diameter"].ewm(alpha=ewm_alpha, min_periods=10).std()
    mu_betw = gf["betw_var"].ewm(alpha=ewm_alpha, min_periods=10).mean()
    sd_betw = gf["betw_var"].ewm(alpha=ewm_alpha, min_periods=10).std()

    z_diam = ((gf["diameter"] - mu_diam) / sd_diam.replace(0, np.nan)).clip(lower=0)
    z_betw = ((gf["betw_var"] - mu_betw) / sd_betw.replace(0, np.nan)).clip(lower=0)
    z_spike = pd.concat([z_diam, z_betw], axis=1).max(axis=1)

    risk = pd.Series(1.0, index=gf.index)
    ramp_zone = (z_spike > z_threshold) & (z_spike <= z_max)
    extreme = z_spike > z_max
    ramp = ((z_spike[ramp_zone] - z_threshold) / (z_max - z_threshold))
    risk[ramp_zone] = 1.0 - ramp * (1.0 - s_min)
    risk[extreme] = s_min

    return risk


# ── Stock scanner ─────────────────────────────────────────


def scan_stock(prices: pd.Series, lookback_days: int = 60) -> dict:
    """Score one stock on regime + risk using its most recent window.

    Returns
    -------
    dict with:
      symbol, n_regimes, latest_regime_size, latest_risk_scale,
      risk_level (normal / warning / danger), regime_label,
      diameter, betw_var
    """
    if len(prices) < lookback_days:
        return {"symbol": getattr(prices, "name", "?"), "error": "not enough data"}

    recent = prices.iloc[-lookback_days:]

    # Regime via Louvain
    communities, node_colors = detect_regimes(recent)
    n_regimes = len(communities)

    # Which regime is the last day in?
    regime_label = f"R{len(communities)}"
    for cid, nodes in enumerate(communities):
        if len(recent) - 1 in nodes:
            regime_label = f"R{cid + 1}"
            break

    # Graph features from the latest window
    G = build_visibility_graph(recent.values)
    metrics = _graph_metrics(G)

    # Risk scale (single value: z-score of diameter + betw_var vs recent history)
    gf = rolling_graph_features(prices, window=lookback_days)
    if not gf.empty:
        risk = compute_risk_scale(gf, ewm_alpha=0.15)
        latest_risk = float(risk.iloc[-1]) if not risk.empty else 1.0
    else:
        latest_risk = 1.0

    if latest_risk >= 0.95:
        risk_level = "normal"
    elif latest_risk >= 0.5:
        risk_level = "warning"
    else:
        risk_level = "danger"

    # Regime trend: is recent regime expanding or contracting?
    regime_sizes = [len(c) for c in communities[-3:]]
    if len(regime_sizes) >= 2 and regime_sizes[-1] > regime_sizes[-2]:
        trend = "expanding"
    elif len(regime_sizes) >= 2:
        trend = "contracting"
    else:
        trend = "stable"

    return {
        "symbol": getattr(prices, "name", "?"),
        "n_regimes": n_regimes,
        "regime_label": regime_label,
        "regime_trend": trend,
        "latest_regime_days": len(communities[-1]),
        "risk_scale": round(latest_risk, 3),
        "risk_level": risk_level,
        "diameter": round(metrics["diameter"], 1),
        "betw_var": round(metrics["betw_var"], 6),
    }


def scan_universe(daily_prices: pd.DataFrame, lookback_days: int = 60) -> pd.DataFrame:
    """Scan every stock in the universe and return a ranked report.

    Returns a DataFrame sorted by risk (most dangerous first).
    """
    rows = []
    for sym in daily_prices.columns:
        try:
            result = scan_stock(daily_prices[sym].dropna(), lookback_days=lookback_days)
            if "error" not in result:
                rows.append(result)
        except Exception as exc:
            logger.warning("Scan skip %s: %s", sym, exc)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("risk_scale")
