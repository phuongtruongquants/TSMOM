"""
Tests for data loading utilities.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from tsmom.data import DEFAULT_SYMBOLS, load_from_csv


class TestLoadFromCsv:
    """Tests for CSV data loading."""

    def test_long_format_csv(self):
        """Load long-format CSV with timestamp, symbol, close columns."""
        csv_content = (
            "timestamp,symbol,close\n"
            "2020-01-01,AAA,10.0\n"
            "2020-01-02,AAA,10.5\n"
            "2020-01-01,BBB,20.0\n"
            "2020-01-02,BBB,21.0\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            tmp_path = f.name

        try:
            df = load_from_csv(tmp_path)
            assert isinstance(df, pd.DataFrame)
            assert set(df.columns) == {"AAA", "BBB"}
            assert df.index.name == "timestamp"
        finally:
            Path(tmp_path).unlink()

    def test_pivoted_format_csv(self):
        """Load already-pivoted CSV (dates as index, symbols as columns)."""
        csv_content = (
            "2020-01-01,10.0,20.0\n"
            "2020-01-02,10.5,21.0\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            tmp_path = f.name

        try:
            df = load_from_csv(tmp_path)
            assert isinstance(df, pd.DataFrame)
        finally:
            Path(tmp_path).unlink()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_from_csv("nonexistent_file.csv")


class TestDefaultSymbols:
    """Tests for the default symbol list."""

    def test_default_symbols_is_list_of_strings(self):
        assert isinstance(DEFAULT_SYMBOLS, list)
        assert all(isinstance(s, str) for s in DEFAULT_SYMBOLS)

    def test_default_symbols_non_empty(self):
        assert len(DEFAULT_SYMBOLS) == 60
