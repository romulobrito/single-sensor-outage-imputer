"""
Tests for CSV/HDF5 table loading and missing PyTables errors.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ssoi.train import load_table


def test_load_table_csv(tmp_path: Path) -> None:
    """
    CSV files load without PyTables.
    """
    path = tmp_path / "frame.csv"
    expected = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    expected.to_csv(path, index=False)
    loaded = load_table(path)
    pd.testing.assert_frame_equal(loaded, expected)


def test_load_table_hdf_key_selects_table(tmp_path: Path) -> None:
    """
    --hdf-key selects one table when the file contains more than one.
    """
    pytest.importorskip("tables")
    path = tmp_path / "two_tables.h5"
    first = pd.DataFrame({"x": [1.0, 2.0]})
    second = pd.DataFrame({"y": [9.0, 8.0, 7.0]})
    first.to_hdf(path, key="first", mode="w")
    second.to_hdf(path, key="second", mode="a")
    loaded = load_table(path, hdf_key="second")
    pd.testing.assert_frame_equal(loaded, second)


def test_load_table_hdf_missing_tables_has_install_hint(
    tmp_path: Path,
) -> None:
    """
    Missing PyTables raises ImportError pointing at the [hdf5] extra.
    """
    path = tmp_path / "needs_tables.h5"
    path.write_bytes(b"not-a-real-hdf5")
    with patch("ssoi.train.pd.read_hdf", side_effect=ImportError("No module named tables")):
        with pytest.raises(ImportError, match=r"\[hdf5\]") as exc_info:
            load_table(path)
    assert 'pip install -e ".[hdf5]"' in str(exc_info.value)
