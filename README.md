# TSMOM — Time-Series Momentum Strategy for Vietnamese Stocks

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An implementation of the **Time-Series Momentum (TSMOM)** strategy applied to the Vietnamese stock market, based on the framework from [Moskowitz, Ooi, and Pedersen (2012)](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf).

## Overview

This project implements a complete TSMOM pipeline:

1. **Data Loading** — Fetch historical price data via `vnstock` / `vnstock_data`, or from a local CSV
2. **Volatility Estimation** — Ex-ante annualized volatility using exponentially weighted variance (COM=60)
3. **Regression Evidence** — Pooled OLS regressions testing momentum predictability across lags
4. **Signal Generation** — Long-only momentum signals from rolling cumulative returns
5. **Backtesting** — Full backtest with volatility-targeted position sizing and transaction costs
6. **Portfolio Analysis** — Equal-weight TSMOM portfolio across 60 stocks, benchmark comparison vs VN-Index

## Project Structure

```
TSMOM/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml            # Package metadata & dev config
├── config.yaml               # Strategy parameters
├── tsmom/
│   ├── __init__.py
│   ├── data.py               # Data loading (vnstock / CSV)
│   ├── volatility.py         # Ex-ante volatility estimation
│   ├── regression.py         # Momentum regression tests
│   ├── signal.py             # Signal generation
│   ├── backtest.py           # Backtesting engine
│   ├── metrics.py            # Performance metrics
│   └── plotting.py           # Visualization
├── scripts/
│   ├── run_backtest.py       # Main backtest pipeline
│   ├── fetch_data.py         # Download & cache data locally
│   └── dashboard.py          # Streamlit interactive dashboard
├── tests/
│   ├── test_data.py
│   ├── test_volatility.py
│   ├── test_signal.py
│   ├── test_backtest.py
│   └── test_metrics.py
├── data/                     # Local data cache (git-ignored)
│   └── .gitkeep
├── output/                   # Charts and reports (git-ignored)
│   └── .gitkeep
├── notebooks/
│   └── TSMOM.ipynb           # Original research notebook
├── docs/
│   ├── project-roadmap.md
│   └── project-changelog.md
└── .gitignore
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

**Option A — Download via vnstock:**
```bash
python scripts/fetch_data.py
```

**Option B — Use your own CSV:**

Place a CSV file at `data/stock_prices.csv` with columns: `timestamp`, `symbol`, `close`.

### 3. Run the full backtest

```bash
python scripts/run_backtest.py
```

This will:
- Load price data for ~60 Vietnamese stocks
- Compute ex-ante volatility
- Run momentum regression tests
- Backtest the TSMOM strategy per-stock and as an equal-weight portfolio
- Generate charts in the `output/` folder

### 4. Explore the interactive dashboard

```bash
streamlit run scripts/dashboard.py
```

Opens a 5-tab interactive dashboard with:
- **Overview** — portfolio metrics, cumulative returns with VN-Index overlay, drawdown
- **Per-Stock** — return heatmap, return distribution histogram
- **Regression Evidence** — TSMOM + sign regression t-stat bar charts (live)
- **TSMOM Smile** — monthly strategy vs benchmark scatter with quadratic fit
- **Volatility** — per-stock vol distribution vs target, time-series vol for selected stock

Adjust parameters via sidebar sliders and charts recompute live.

### 5. Customize parameters

Edit `config.yaml`:

```yaml
strategy:
  vol_target: 0.40        # Annualized volatility target
  lookback_window: 6      # Weeks of lookback for momentum signal
  commission: 0.001       # One-way transaction cost
  margin_cap: 2.0         # Maximum leverage
  ewm_com: 60             # EWM center-of-mass for volatility

data:
  source: "csv"           # "csv" or "vnstock"
  csv_path: "data/stock_prices.csv"
  start_date: "2014-01-01"
  end_date: "2025-03-20"
```

## Key Concepts

### Ex-Ante Volatility

$$\sigma_t = \sqrt{252 \cdot \text{EWM}\_\text{var}(r_{t-1}, \text{com}=60)}$$

### Momentum Regression

$$(r_{t+h} / \sigma_{t+h-1}) = \alpha + \beta_h \cdot (r_t / \sigma_{t-1}) + \varepsilon$$

### Position Sizing

$$\text{position}_t = \text{signal}_t \cdot \min\left(\frac{\sigma^*}{\hat\sigma_t},\; 2\right)$$

where $\sigma^* = 0.40$ is the volatility target.

## Output

The pipeline generates:
- **Per-stock backtest** with cumulative returns, position sizes, and signals
- **Regression t-stat charts** by lag
- **Portfolio cumulative return** chart
- **Return / volatility distribution** histograms
- **TSMOM vs VN-Index scatter** with quadratic fit
- **Summary metrics table** (annualized return, Sharpe, skewness, kurtosis)

## References

- Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012). *Time Series Momentum*. Journal of Financial Economics.
- Baltas, A.-N. & Kosowski, R. (2013). *Momentum Strategies in Futures Markets and Trend-following Funds*.

## License

MIT — see [LICENSE](LICENSE).
