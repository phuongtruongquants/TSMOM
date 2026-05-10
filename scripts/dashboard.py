#!/usr/bin/env python3
"""
Streamlit dashboard for TSMOM strategy analysis.

Usage:
    streamlit run scripts/dashboard.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsmom.data import load_data, load_benchmark_csv
from tsmom.volatility import compute_weekly_volatility, exante_volatility
from tsmom.regression import run_tsmom_regressions, run_sign_regressions
from tsmom.backtest import backtest_universe, summarize_universe_returns
from tsmom.metrics import calculate_metrics, summarize_metrics

st.set_page_config(
    page_title="TSMOM — Vietnam Momentum Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("TSMOM Strategy Controls")

config_path = Path("config.yaml")
if config_path.exists():
    cfg = yaml.safe_load(config_path.read_text())
else:
    cfg = {}

strat_defaults = cfg.get("strategy", {})

vol_target = st.sidebar.slider(
    "Volatility Target", 0.1, 0.8, float(strat_defaults.get("vol_target", 0.4)), 0.05,
    help="Annualized volatility target for position sizing.",
)
lookback = st.sidebar.slider(
    "Lookback Window (weeks)", 2, 52, int(strat_defaults.get("lookback_window", 6)), 1,
    help="Rolling weeks for momentum signal.",
)
commission = st.sidebar.slider(
    "Commission (bps)", 0.0, 0.01, float(strat_defaults.get("commission", 0.001)), 0.0005,
    format="%.4f",
    help="One-way transaction cost.",
)
margin_cap = st.sidebar.slider(
    "Margin Cap", 1.0, 5.0, float(strat_defaults.get("margin_cap", 2.0)), 0.5,
    help="Maximum leverage multiple.",
)
ewm_com = st.sidebar.slider(
    "EWM Center-of-Mass", 20, 120, int(strat_defaults.get("ewm_com", 60)), 10,
    help="Decay rate for volatility estimation.",
)

plot_dpi = st.sidebar.slider("Chart DPI", 100, 300, 150, 25)

# ── Data loading ─────────────────────────────────────────
st.title("TSMOM — Time-Series Momentum for Vietnamese Stocks")

@st.cache_data(ttl=3600)
def load_all_data():
    cfg_copy = dict(cfg)
    cfg_copy.setdefault("data", {})["source"] = "csv"
    cfg_copy["data"]["csv_path"] = "data/stock_prices.csv"
    prices = load_data(cfg_copy)
    bench = None
    bench_path = cfg_copy["data"].get("benchmark_csv", "data/vni.csv")
    if Path(bench_path).exists():
        bench = load_benchmark_csv(bench_path)
    return prices, bench

status = st.empty()
try:
    with st.spinner("Loading data..."):
        daily_prices, benchmark = load_all_data()
    status.success(f"Loaded {len(daily_prices.columns)} stocks × {len(daily_prices)} days")
except FileNotFoundError as e:
    status.error(f"Cannot load data: {e}")
    st.stop()

# ── Compute ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def run_backtest(_daily_prices, _vol_target, _lookback, _commission, _margin_cap, _ewm_com):
    cpu = _daily_prices.copy()  # break cache ref
    all_returns, all_metrics = backtest_universe(
        cpu,
        vol_target=_vol_target,
        lookback=_lookback,
        commission=_commission,
        margin_cap=_margin_cap,
        ewm_com=_ewm_com,
    )
    port_ret, port_metrics = summarize_universe_returns(all_returns)
    return all_returns, all_metrics, port_ret, port_metrics


@st.cache_data(ttl=300)
def run_stability_sweep(_daily_prices, _commission, _margin_cap):
    cpu = _daily_prices.copy()
    rows = []
    for lookback_value in [4, 6, 8, 12]:
        for vol_target_value in [0.2, 0.4, 0.6]:
            for ewm_com_value in [40, 60, 80]:
                all_returns, _ = backtest_universe(
                    cpu,
                    vol_target=vol_target_value,
                    lookback=lookback_value,
                    commission=_commission,
                    margin_cap=_margin_cap,
                    ewm_com=ewm_com_value,
                )
                port_ret, _ = summarize_universe_returns(all_returns)
                summary = summarize_metrics(port_ret)
                rows.append({
                    "lookback": lookback_value,
                    "vol_target": vol_target_value,
                    "ewm_com": ewm_com_value,
                    **summary,
                })
    return pd.DataFrame(rows)


with st.spinner("Running backtest..."):
    all_returns, all_metrics, port_ret, port_metrics = run_backtest(
        daily_prices, vol_target, lookback, commission, margin_cap, ewm_com,
    )

# ── Tabs ─────────────────────────────────────────────────
tab_overview, tab_perstock, tab_regression, tab_smile, tab_vol, tab_stability = st.tabs([
    "Overview", "Per-Stock", "Regression Evidence", "TSMOM Smile", "Volatility", "Stability",
])

# ── Tab 1: Overview ──────────────────────────────────────
with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Annualized Return", f"{port_metrics.loc['Annualized Return', 'metrics']:.2%}")
    col2.metric("Sharpe Ratio", f"{port_metrics.loc['Sharpe Ratio', 'metrics']:.2f}")
    col3.metric("Max Drawdown", f"{port_metrics.loc['Max Drawdown', 'metrics']:.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Ann. Volatility", f"{port_metrics.loc['Annualized Volatility', 'metrics']:.2%}")
    col5.metric("Skewness", f"{port_metrics.loc['Skewness', 'metrics']:.2f}")
    col6.metric("Kurtosis", f"{port_metrics.loc['Kurtosis', 'metrics']:.2f}")

    st.subheader("Portfolio Cumulative Return")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=plot_dpi)
    cum_tsmom = (1 + port_ret).cumprod()
    cum_tsmom.plot(ax=ax, color="royalblue", lw=2, label="TSMOM (equal-weight)")

    # Benchmark overlay
    if benchmark is not None and "close" in benchmark.columns:
        bm_weekly = benchmark.resample("W").last().ffill()
        bm_weekly["ret"] = bm_weekly["close"].pct_change()
        cum_bm = (1 + bm_weekly["ret"]).cumprod()
        cum_bm = cum_bm.reindex(cum_tsmom.index).ffill()
        cum_bm.plot(ax=ax, color="orange", lw=1.5, ls="--", label="VN-Index")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Drawdown
    st.subheader("Drawdown")
    cum = (1 + port_ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    fig, ax = plt.subplots(figsize=(12, 3), dpi=plot_dpi)
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.3)
    ax.plot(dd.index, dd.values, color="crimson", lw=1)
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 2: Per-Stock ─────────────────────────────────────
with tab_perstock:
    st.subheader("Per-Stock Return Heatmap")

    # Filter to stocks with metrics
    if "Annualized Return" in all_metrics.index:
        st.write(f"Showing returns for {all_metrics.shape[1]} stocks")

        # Heatmap of cumulative per-stock returns over time (sampled quarterly)
        cum_rets = (1 + all_returns).cumprod()
        display_years = st.slider(
            "Display years", 1, len(cum_rets.resample("YE").last()),
            min(5, len(cum_rets.resample("YE").last())),
            key="perstock_years",
        )
        quarterly = cum_rets.resample("QE").last().tail(display_years * 4 + 1)

        if not quarterly.empty:
            fig, ax = plt.subplots(figsize=(14, max(6, len(quarterly) * 0.5)), dpi=plot_dpi)
            im = ax.imshow(quarterly.T.values, aspect="auto", cmap="RdYlGn",
                          vmin=0.5, vmax=2.0, interpolation="nearest")
            ax.set_yticks(range(len(quarterly.columns)))
            ax.set_yticklabels(quarterly.columns, fontsize=7)
            xtick_idx = np.linspace(0, len(quarterly) - 1, min(10, len(quarterly)), dtype=int)
            ax.set_xticks(xtick_idx)
            ax.set_xticklabels([quarterly.index[i].strftime("%Y-%m") for i in xtick_idx],
                              rotation=45, ha="right")
            ax.set_title("Cumulative Return by Stock (green = profit, red = loss)")
            plt.colorbar(im, ax=ax, shrink=0.8, label="Cumulative Return")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Return Distribution")
    if "Annualized Return" in all_metrics.index:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=plot_dpi)
        rets = all_metrics.loc["Annualized Return"].dropna().astype(float)
        ax.hist(rets, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        ax.axvline(rets.mean(), color="red", ls="--", label=f"Mean: {rets.mean():.2%}")
        ax.set_xlabel("Annualized Return")
        ax.set_ylabel("Number of Stocks")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 3: Regression Evidence ────────────────────────────
with tab_regression:
    st.subheader("Momentum Regression — T-statistics by Lag")

    max_lag = st.slider("Max Lag", 4, 50, 24, key="reg_max_lag")

    with st.spinner("Running regressions..."):
        weekly_prices = daily_prices.resample("W").last().ffill()
        weekly_ret = weekly_prices.pct_change().dropna()
        weekly_vol = compute_weekly_volatility(daily_prices, com=ewm_com)

        dummy_vol = exante_volatility(
            daily_prices.iloc[:, 0].pct_change().dropna(), com=ewm_com
        )
        weekly_vol = weekly_vol.reindex(index=weekly_ret.index).ffill()

        tsmom_reg = run_tsmom_regressions(weekly_ret, weekly_vol, max_lag=max_lag)
        sign_reg = run_sign_regressions(weekly_ret, weekly_vol, max_lag=max_lag)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=plot_dpi)

    lags = range(1, len(tsmom_reg) + 1)
    ax1.bar(lags, tsmom_reg["tstat"], color="skyblue", edgecolor="black")
    ax1.axhline(y=0, color="black", lw=1)
    ax1.axhline(y=1.96, color="red", ls="--", lw=1)
    ax1.axhline(y=-1.96, color="red", ls="--", lw=1)
    ax1.set_title("TSMOM Regression")
    ax1.set_xlabel("Week Lag")
    ax1.set_ylabel("T-statistic")
    ax1.set_xticks(list(lags)[::max(1, max_lag // 12)])
    ax1.grid(True, linestyle="--", alpha=0.5)

    slags = range(1, len(sign_reg) + 1)
    ax2.bar(slags, sign_reg["tstat"], color="lightcoral", edgecolor="black")
    ax2.axhline(y=0, color="black", lw=1)
    ax2.axhline(y=1.96, color="red", ls="--", lw=1)
    ax2.axhline(y=-1.96, color="red", ls="--", lw=1)
    ax2.set_title("Sign Regression")
    ax2.set_xlabel("Week Lag")
    ax2.set_ylabel("T-statistic")
    ax2.set_xticks(list(slags)[::max(1, max_lag // 12)])
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 4: TSMOM Smile ────────────────────────────────────
with tab_smile:
    st.subheader("TSMOM Returns vs VN-Index Returns")

    if benchmark is not None and "close" in benchmark.columns:
        bench_weekly = benchmark.resample("W").last().ffill()
        bench_weekly["vni"] = bench_weekly["close"].pct_change()

        combined = pd.DataFrame({
            "tsmom": port_ret,
            "vni": bench_weekly["vni"],
        }).dropna()

        if not combined.empty:
            monthly = combined.resample("ME").apply(lambda x: (1 + x).prod() - 1)

            x = monthly["vni"]
            y = monthly["tsmom"]

            X = pd.DataFrame({"x": x, "x_sq": x ** 2})
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            x_fit = np.linspace(x.min(), x.max(), 100)
            X_fit = sm.add_constant(pd.DataFrame({"x": x_fit, "x_sq": x_fit ** 2}))
            y_fit = model.predict(X_fit)

            fig, ax = plt.subplots(figsize=(8, 7), dpi=plot_dpi)
            ax.scatter(x, y, color="skyblue", edgecolor="black", alpha=0.7, s=40)
            ax.plot(x_fit, y_fit, color="black", ls="--", lw=2, label="Quadratic Fit")
            ax.axhline(y=0, color="gray", lw=0.5)
            ax.axvline(x=0, color="gray", lw=0.5)
            ax.set_xlabel("VN-Index Monthly Return")
            ax.set_ylabel("TSMOM Monthly Return")
            ax.set_title("The TSMOM 'Smile'")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                f"Quadratic coefficient (x²): {model.params.get('x_sq', 0):.4f} "
                f"(t={model.tvalues.get('x_sq', 0):.2f}) — "
                + ("Convexity detected ✓" if model.params.get("x_sq", 0) > 0
                   else "No convexity — flat or inverted")
            )
        else:
            st.warning("Not enough overlapping data to construct TSMOM vs VNI scatter.")
    else:
        st.info("No benchmark data available. Add data/vni.csv for the smile chart.")

# ── Tab 5: Volatility ─────────────────────────────────────
with tab_vol:
    st.subheader("Ex-Ante Volatility Distribution")

    daily_rets = daily_prices.pct_change().dropna()
    latest_vols = {}
    for sym in daily_prices.columns[:30]:  # top 30 for readability
        try:
            vols = exante_volatility(daily_rets[sym].dropna(), com=ewm_com)
            if not vols.empty and vols.iloc[-1] > 0:
                latest_vols[sym] = vols.iloc[-1]
        except Exception:
            pass

    if latest_vols:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=plot_dpi)
        sorted_items = sorted(latest_vols.items(), key=lambda x: x[1])
        symbols, vols = zip(*sorted_items)
        colors = ["green" if v < vol_target else "red" for v in vols]
        ax.bar(range(len(symbols)), vols, color=colors, edgecolor="black", alpha=0.7)
        ax.axhline(y=vol_target, color="black", ls="--", lw=1.5,
                    label=f"Target ({vol_target:.0%})")
        ax.set_xticks(range(len(symbols)))
        ax.set_xticklabels(symbols, rotation=90, fontsize=8)
        ax.set_ylabel("Annualized Volatility")
        ax.set_title("Latest Ex-Ante Volatility by Stock")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Volatility over time for selected stock
    st.subheader("Volatility Over Time")
    selected_sym = st.selectbox("Select stock", list(daily_prices.columns),
                                 index=list(daily_prices.columns).index("HPG")
                                 if "HPG" in daily_prices.columns else 0,
                                 key="vol_stock")

    if selected_sym:
        vol_series = exante_volatility(
            daily_prices[selected_sym].pct_change().dropna(),
            com=ewm_com,
        )
        fig, ax = plt.subplots(figsize=(12, 4), dpi=plot_dpi)
        ax.plot(vol_series, color="royalblue", lw=1)
        ax.axhline(y=vol_target, color="red", ls="--", lw=1, alpha=0.7)
        ax.set_ylabel("Volatility")
        ax.set_title(f"Ex-Ante Volatility — {selected_sym}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 6: Stability ──────────────────────────────────────
with tab_stability:
    st.subheader("Parameter Stability Lab")
    st.caption("Check whether performance holds across nearby settings instead of one lucky parameter choice.")

    with st.spinner("Running parameter sweep..."):
        stability = run_stability_sweep(daily_prices, commission, margin_cap)

    if stability.empty:
        st.warning("No stability results available for the current dataset.")
    else:
        metric_options = [
            "Sharpe Ratio",
            "Annualized Return",
            "Max Drawdown",
            "Positive Weeks",
            "Annualized Volatility",
        ]
        selected_metric = st.selectbox("Metric", metric_options, index=0)
        held_ewm = st.selectbox(
            "Hold EWM Center-of-Mass constant",
            sorted(stability["ewm_com"].unique()),
            index=1,
        )

        filtered = stability[stability["ewm_com"] == held_ewm].copy()
        heatmap = filtered.pivot(index="lookback", columns="vol_target", values=selected_metric)

        if not heatmap.empty:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=plot_dpi)
            values = heatmap.values.astype(float)
            cmap = "RdYlGn" if selected_metric != "Max Drawdown" else "RdYlGn_r"
            im = ax.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
            ax.set_xticks(range(len(heatmap.columns)))
            ax.set_xticklabels([f"{value:.1f}" for value in heatmap.columns])
            ax.set_yticks(range(len(heatmap.index)))
            ax.set_yticklabels([str(value) for value in heatmap.index])
            ax.set_xlabel("Volatility Target")
            ax.set_ylabel("Lookback (weeks)")
            ax.set_title(f"{selected_metric} across parameter settings")

            for row_idx in range(values.shape[0]):
                for col_idx in range(values.shape[1]):
                    value = values[row_idx, col_idx]
                    if pd.notna(value):
                        if selected_metric in {"Annualized Return", "Max Drawdown", "Positive Weeks", "Annualized Volatility"}:
                            label = f"{value:.1%}"
                        else:
                            label = f"{value:.2f}"
                        ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)

            plt.colorbar(im, ax=ax, shrink=0.85, label=selected_metric)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Median Sharpe", f"{stability['Sharpe Ratio'].median():.2f}")
        summary_col2.metric("Positive Sharpe %", f"{(stability['Sharpe Ratio'] > 0).mean():.0%}")
        summary_col3.metric("Positive Return %", f"{(stability['Annualized Return'] > 0).mean():.0%}")
        summary_col4.metric("Sharpe IQR", f"{stability['Sharpe Ratio'].quantile(0.75) - stability['Sharpe Ratio'].quantile(0.25):.2f}")

        baseline_mask = (
            (stability["lookback"] == lookback)
            & np.isclose(stability["vol_target"], vol_target)
            & (stability["ewm_com"] == ewm_com)
        )
        baseline_row = stability[baseline_mask]
        if not baseline_row.empty:
            best_sharpe = stability["Sharpe Ratio"].max()
            baseline_sharpe = baseline_row.iloc[0]["Sharpe Ratio"]
            st.caption(
                f"Best sweep Sharpe: {best_sharpe:.2f}. Current setting Sharpe: {baseline_sharpe:.2f}. "
                f"Gap: {best_sharpe - baseline_sharpe:.2f}."
            )
        else:
            st.caption("Current sidebar parameters sit outside the compact sweep grid, so the comparison uses grid results only.")

        ranked = stability.copy()
        if selected_metric == "Max Drawdown":
            ranked = ranked.sort_values([selected_metric, "Sharpe Ratio"], ascending=[False, False])
        elif selected_metric == "Annualized Volatility":
            ranked["Vol Target Gap"] = (ranked["Annualized Volatility"] - ranked["vol_target"]).abs()
            ranked = ranked.sort_values(["Vol Target Gap", "Sharpe Ratio"], ascending=[True, False])
        else:
            ranked = ranked.sort_values([selected_metric, "Sharpe Ratio"], ascending=[False, False])
        display = ranked[["lookback", "vol_target", "ewm_com", "Sharpe Ratio", "Annualized Return", "Max Drawdown", "Positive Weeks"]].copy()
        display["vol_target"] = display["vol_target"].map(lambda value: f"{value:.1f}")
        display["Annualized Return"] = display["Annualized Return"].map(lambda value: f"{value:.2%}")
        display["Max Drawdown"] = display["Max Drawdown"].map(lambda value: f"{value:.2%}")
        display["Positive Weeks"] = display["Positive Weeks"].map(lambda value: f"{value:.1%}")

        st.markdown("**Top settings**")
        st.dataframe(display.head(10), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    "TSMOM strategy based on [Moskowitz, Ooi & Pedersen (2012)]"
    "(https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf).\n"
    "Data source: Vietnamese stock market via vnstock."
)
