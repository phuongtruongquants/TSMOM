"""
Visualization functions for TSMOM analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def plot_tstats_by_lag(tstats, title="T-statistic by Week Lag", save_path=None, dpi=200):
    """Bar chart of regression t-stats by lag with 95% CI lines."""
    lags = range(1, len(tstats) + 1)
    fig, ax = plt.subplots(figsize=(15, 5), dpi=dpi)
    ax.bar(lags, tstats, color="skyblue", edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.axhline(y=1.96, color="red", linestyle="--", linewidth=1, label="95% CI")
    ax.axhline(y=-1.96, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Week Lag")
    ax.set_ylabel("T-statistic of Beta")
    ax.set_title(title)
    ax.set_xticks(list(lags))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_single_backtest(strategy, symbol, save_path=None, dpi=200):
    """3-panel chart: returns, cumulative returns, and position sizing."""
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), dpi=dpi)

    ax[0].plot(strategy["raw_rets"], label="Raw Returns")
    ax[0].plot(strategy["strat_rets"], label="Strategy Returns", linestyle="--")
    ax[0].set_title(f"{symbol} Strategy Returns")
    ax[0].legend()

    ax[1].plot((1 + strategy["raw_rets"]).cumprod(), label="Buy & Hold")
    ax[1].plot((1 + strategy["strat_rets"]).cumprod(), label="TSMOM", linestyle="--")
    ax[1].set_title(f"{symbol} Cumulative Returns")
    ax[1].legend()

    ax[2].plot(strategy["position_size"], label="Position Size")
    ax[2].set_title(f"{symbol} Position Size")
    ax2 = ax[2].twinx()
    ax2.plot(strategy["signal"], color="red", linestyle="--", label="Signal")
    ax[2].legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_portfolio_cumret(returns, title=None, save_path=None, dpi=200):
    """Cumulative return of the equal-weight TSMOM portfolio."""
    cum = (1 + returns.mean(axis=1)).cumprod()
    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    cum.plot(ax=ax, color="royalblue", lw=2)
    ax.set_title(title or "TSMOM Portfolio — Vietnam Stock Market", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Cumulative Return", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_distribution(data, title, xlabel, save_path=None, dpi=200):
    """Histogram of a metric distribution across stocks."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.hist(data, bins=15, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_tsmom_vs_benchmark(tsmom_rets, bench_rets, save_path=None, dpi=200):
    """Scatter + quadratic fit of TSMOM monthly returns vs VN-Index."""
    combined = pd.DataFrame({"tsmom": tsmom_rets, "vni": bench_rets}).dropna()
    if combined.empty:
        return None
    monthly = combined.resample("M").apply(lambda x: (1 + x).prod() - 1)

    x = monthly["vni"]
    y = monthly["tsmom"]

    X = pd.DataFrame({"x": x, "x_sq": x ** 2})
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    x_fit = np.linspace(x.min(), x.max(), 100)
    X_fit = sm.add_constant(pd.DataFrame({"x": x_fit, "x_sq": x_fit ** 2}))
    y_fit = model.predict(X_fit)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.scatter(x, y, color="skyblue", edgecolor="black", alpha=0.7, label="Data")
    ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=1.5, label="Quadratic Fit")
    ax.set_title("TSMOM vs VN-Index", fontsize=16)
    ax.set_xlabel("VN-Index Returns", fontsize=14)
    ax.set_ylabel("TSMOM Returns", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_volatility_price(vol_series, price_series, symbol, save_path=None, dpi=200):
    """Dual-axis plot of volatility vs price for a single stock."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    color1, color2 = "tab:blue", "tab:orange"

    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility", color=color1)
    ax.plot(vol_series, color=color1, label="Volatility")
    ax.tick_params(axis="y", labelcolor=color1)
    ax.grid(True, linestyle="--", alpha=0.5, color="lightgray")

    ax2 = ax.twinx()
    ax2.set_ylabel("Price", color=color2)
    ax2.plot(price_series, color=color2, linestyle="--", label="Price")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax.set_title(f"Volatility & Price — {symbol}", fontsize=14)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2,
               loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
