# Project Changelog

## [1.1.0] — 2026-05-11

### Added
- **Streamlit dashboard** (`scripts/dashboard.py`) — 5-tab interactive UI: overview metrics, per-stock heatmap, regression evidence, TSMOM smile, volatility analysis. Live recomputation on slider changes.
- **Test suite** — 26 tests across 5 files (`test_volatility.py`, `test_signal.py`, `test_backtest.py`, `test_metrics.py`, `test_data.py`). All pass.
- **`pyproject.toml`** — full metadata, build config, dev dependencies, ruff + pytest config.
- **`vnstock` + `streamlit`** added to `requirements.txt`.

### Fixed
- **Look-ahead bug** in `backtest.py` — `position * weekly_return.shift(-1)` → `position.shift(1) * weekly_return`. Trade costs also realigned.
- **Bare `except Exception`** in `backtest_universe` → `except (ValueError, KeyError, IndexError)`.
- **Exception chaining** in `data.py` — `raise ... from exc` properly threaded in vnstock import fallback.

### Changed
- Ruff config with per-file exceptions for stats notation (N806 in `regression.py`, `plotting.py`).

---

## [1.0.0] — Initial Release

- TSMOM backtesting engine with volatility-targeted position sizing
- Momentum regression tests (TSMOM + sign)
- Signal generation (long-only, rolling cumulative returns)
- VN-Index benchmark comparison
- Data loading: CSV + vnstock backends
- 60-stock default universe (HOSE-listed)
