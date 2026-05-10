# TSMOM — Time-Series Momentum for Vietnamese Stocks

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-28%20passed-green.svg)](tests/)
[![Ruff](https://img.shields.io/badge/lint-ruff%20clean-blue.svg)](pyproject.toml)

Implementation of the **Time-Series Momentum (TSMOM)** strategy applied to 60 Vietnamese stocks, based on [Moskowitz, Ooi & Pedersen (2012)](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf).

## Live Dashboard

**[ptqtsmom.streamlit.app](https://ptqtsmom.streamlit.app)** — interactive 6-tab dashboard with live parameter controls.

## Results (2014–2026, 60 HOSE stocks)

| Metric | TSMOM Portfolio |
|--------|:--------------:|
| Annualized Return | 15.69% |
| Annualized Volatility | 12.21% |
| Sharpe Ratio | 1.28 |
| Max Drawdown | -16.54% |
| Positive Weeks | 59.5% |
| Momentum t-stat (lag 1) | 11.47 |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run backtest pipeline
python scripts/run_backtest.py

# Launch interactive dashboard
streamlit run scripts/dashboard.py
```

The repo ships with 60 stocks of daily price data (`data/stock_prices.csv`). To refresh it:

```bash
pip install vnstock_data
python scripts/fetch_data.py --start 2014-01-01 --end 2026-05-11
```

## Dashboard Tabs

| Tab | What it shows |
|-----|--------------|
| Overview | Portfolio metrics, cumulative return vs VN-Index, drawdown |
| Per-Stock | Return heatmap, distribution histogram |
| Regression Evidence | TSMOM + sign regression t-stat charts (live) |
| TSMOM Smile | Monthly strategy vs benchmark scatter with quadratic fit |
| Volatility | Per-stock vol vs target, time-series for selected stock |
| Stability | Parameter sweep across lookback × vol target × EWM com |

Sliders update all charts in real time.

## Project Structure

```
TSMOM/
├── tsmom/                  # Core library
│   ├── data.py             # CSV / vnstock_data loading
│   ├── volatility.py       # Ex-ante vol (EWM, com=60)
│   ├── regression.py       # Pooled OLS momentum tests
│   ├── signal.py           # Long-only momentum signal
│   ├── backtest.py         # Per-stock & universe backtester
│   ├── metrics.py          # Sharpe, drawdown, skew, kurtosis
│   └── plotting.py         # Matplotlib charts
├── scripts/
│   ├── dashboard.py        # Streamlit (6 tabs, live controls)
│   ├── run_backtest.py     # CLI pipeline
│   └── fetch_data.py       # Download from vnstock_data
├── tests/                  # pytest (28 tests)
├── data/stock_prices.csv   # 60 stocks, 2014–2026
├── config.yaml             # Strategy parameters
├── pyproject.toml          # Package & dev config
└── docs/                   # Roadmap & changelog
```

## Methodology

### Signal
$$S_t = \max\!\left(0,\ \mathrm{sign}\!\left(\prod_{i=t-W}^{t} (1 + r_i) - 1\right)\right)$$
where $W$ is the lookback window (default 6 weeks). Long-only — negative momentum signals are clipped to 0.

### Ex-Ante Volatility
$$\sigma_t = \sqrt{252 \cdot \mathrm{EWM}_{\mathrm{var}}(r_{t-1}, \mathrm{com}=60)}$$

### Position Sizing
$$\displaystyle \mathrm{pos}_t = S_t \cdot \min\!\left(\frac{\sigma_{\mathrm{target}}}{\hat\sigma_t},\ 2\right),\quad \sigma_{\mathrm{target}} = 0.40$$

### Regression
$$r_{t+h} / \sigma_{t+h-1} = \alpha + \beta_h \cdot (r_t / \sigma_{t-1}) + \varepsilon$$

## Customize

Edit `config.yaml`:

```yaml
strategy:
  vol_target: 0.40       # Annualized vol target
  lookback_window: 6     # Momentum lookback (weeks)
  commission: 0.001      # One-way transaction cost
  margin_cap: 2.0        # Max leverage
  ewm_com: 60            # EWM center-of-mass
```

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v   # 28 passed
python -m ruff check tsmom/  # clean
```

## References

- Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012). *Time Series Momentum*. Journal of Financial Economics.
- Baltas, A.-N. & Kosowski, R. (2013). *Momentum Strategies in Futures Markets and Trend-following Funds*.

## License

MIT — see [LICENSE](LICENSE).
