# Project Roadmap

## Current Status

**Version:** 1.1.0
**Phase:** Polish & Ship — done
**Last updated:** 2026-05-11

---

## Phase 1 — Core (done)

- [x] Data loading (CSV + vnstock)
- [x] Ex-ante volatility estimation (EWM)
- [x] Momentum signal generation (long-only)
- [x] Per-stock backtest engine
- [x] Universe backtest (equal-weight portfolio)
- [x] Performance metrics (Sharpe, max DD, skew, kurtosis)
- [x] Regression evidence (TSMOM + sign)
- [x] Plotting utilities
- [x] Config-driven (yaml)
- [x] `requirements.txt` + `pyproject.toml`

## Phase 2 — Quality (done)

- [x] Fix look-ahead bug in backtest
- [x] Fix bare exception handling
- [x] Test suite (26 tests, all pass)
- [x] Ruff linting (clean)
- [x] Streamlit dashboard with live controls

## Phase 3 — Strategy Depth (next)

- [ ] Multi-horizon signals (1/3/6/12-month combined)
- [ ] Long/short version alongside long-only
- [ ] Bootstrapped Sharpe CIs
- [ ] Walk-forward parameter sweep
- [ ] Sensitivity curves (Sharpe vs commission, vs vol target, vs lookback)

## Phase 4 — Data Quality

- [ ] Point-in-time index membership (survivorship bias fix)
- [ ] Total returns (dividends + splits)
- [ ] Foreign-flow signal (Vietnam-specific edge)

## Phase 5 — Production

- [ ] Dockerfile
- [ ] GitHub Actions CI (pytest + ruff)
- [ ] DuckDB data layer (replace CSV)
- [ ] `vectorbt` integration (faster sweeps)

## Success Metrics

| Metric | Baseline | Current |
|--------|----------|---------|
| Tests | 0 | 26 pass |
| Lint errors | ~41 | 0 |
| Dashboard | none | 5-tab interactive |
| Installable | manual | `pip install -e .` |
