"""
Unit tests for leakage-safe split, numeric target cleanup, and outage channels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ssoi.train import (
    _outage_channels,
    blocked_split_with_skip,
    coerce_numeric_series,
)


def _split_kwargs(**overrides: object) -> dict[str, object]:
    """Defaults small enough for synthetic tables."""
    params: dict[str, object] = {
        "test_size": 0.05,
        "val_size": 0.1,
        "min_target_coverage": 0.2,
        "min_target_count": 5,
        "auto_skip_until_coverage": True,
        "max_skip_head_frac": 0.95,
    }
    params.update(overrides)
    return params


def test_duplicate_index_split_uses_row_positions() -> None:
    """
    Duplicate index labels must not pull extra rows into the train block.
    """
    n = 30
    values = np.arange(n, dtype=np.float64)
    df = pd.DataFrame({"y": values}, index=list(range(3)) * 10)
    train_pos, val_pos, test_pos, _frac = blocked_split_with_skip(
        df,
        target="y",
        **_split_kwargs(min_target_coverage=0.0, min_target_count=1),
    )
    assert len(df.iloc[train_pos]) == len(train_pos)
    labels = df.index.to_numpy()[train_pos]
    leaked = df.loc[labels]
    assert len(leaked) > len(train_pos)
    np.testing.assert_array_equal(train_pos, np.arange(len(train_pos)))
    selected = set(train_pos.tolist() + val_pos.tolist() + test_pos.tolist())
    assert len(selected) == len(train_pos) + len(val_pos) + len(test_pos)


def test_dirty_target_tokens_are_not_observed() -> None:
    """
    Invalid strings must not count as observed targets in coverage.
    """
    raw = pd.Series(["1.0", "bad", None, "--"])
    cleaned = coerce_numeric_series(raw)
    assert cleaned.notna().tolist() == [True, False, False, False]
    assert float(cleaned.iloc[0]) == pytest.approx(1.0)

    n_good = 25
    y = ["bad", "--", None, "null"] * 5 + [1.5] * n_good
    df = pd.DataFrame({"y": y})
    train_pos, _val, _test, frac = blocked_split_with_skip(
        df,
        target="y",
        **_split_kwargs(min_target_coverage=0.8, min_target_count=15),
    )
    observed_train = coerce_numeric_series(df["y"]).iloc[train_pos].notna().sum()
    assert int(observed_train) >= 15
    assert frac > 0.0


def test_infinite_target_is_treated_as_missing() -> None:
    """
    +/-inf must become NaN before coverage and training filters.
    """
    series = coerce_numeric_series(pd.Series([1.0, np.inf, -np.inf, 2.0]))
    assert series.isna().tolist() == [False, True, True, False]

    y = [np.inf] * 20 + [1.0] * 20
    df = pd.DataFrame({"y": y})
    train_pos, _val, _test, frac = blocked_split_with_skip(
        df,
        target="y",
        **_split_kwargs(min_target_coverage=0.9, min_target_count=10),
    )
    observed = coerce_numeric_series(df["y"]).iloc[train_pos]
    assert int(observed.notna().sum()) >= 10
    assert bool(np.isfinite(observed.dropna().to_numpy()).all())
    assert frac > 0.0


def test_head_skip_raises_when_no_window_meets_thresholds() -> None:
    """
    Adaptive head-skip must fail loudly instead of returning the initial split.
    """
    df = pd.DataFrame({"y": np.full(40, np.nan)})
    with pytest.raises(ValueError, match="min_target_coverage") as exc_info:
        blocked_split_with_skip(
            df,
            target="y",
            **_split_kwargs(min_target_coverage=0.2, min_target_count=10),
        )
    message = str(exc_info.value)
    assert "best coverage=" in message.lower() or "Best coverage=" in message
    assert "best observed count=" in message
    assert "max skip_frac tested=" in message


def test_outage_channels_are_ones_and_zeros() -> None:
    """
    Production selection channels are m_t=1 and t_tilde=0.
    """
    yb = torch.tensor([[1.5], [-0.25], [3.0]], dtype=torch.float32)
    t_tilde, m_t = _outage_channels(yb)
    assert t_tilde.shape == yb.shape
    assert m_t.shape == yb.shape
    assert torch.equal(m_t, torch.ones_like(yb))
    assert torch.equal(t_tilde, torch.zeros_like(yb))


def test_happy_split_is_contiguous_and_disjoint() -> None:
    """
    Numeric target with low thresholds yields contiguous blocked folds.
    """
    n = 40
    df = pd.DataFrame({"y": np.linspace(0.0, 1.0, n)})
    train_pos, val_pos, test_pos, frac = blocked_split_with_skip(
        df,
        target="y",
        **_split_kwargs(min_target_coverage=0.2, min_target_count=5),
    )
    assert frac == pytest.approx(0.0)
    assert train_pos[-1] + 1 == val_pos[0]
    assert val_pos[-1] + 1 == test_pos[0]
    assert test_pos[-1] == n - 1
    train_set = set(train_pos.tolist())
    val_set = set(val_pos.tolist())
    test_set = set(test_pos.tolist())
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
