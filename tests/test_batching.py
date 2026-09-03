"""
Tests for BatchNorm last-batch handling.
"""
from __future__ import annotations

import pytest

from ssoi.train import _drop_last_for_batchnorm


def test_drop_last_when_remainder_is_one() -> None:
    """Last batch of size 1 must be dropped in train mode."""
    assert _drop_last_for_batchnorm(257, 256) is True


def test_drop_last_false_when_full_batches() -> None:
    """Exact multiples keep every sample."""
    assert _drop_last_for_batchnorm(256, 256) is False


def test_drop_last_false_when_remainder_gt_one() -> None:
    """A remainder larger than 1 is a valid BatchNorm batch."""
    assert _drop_last_for_batchnorm(255, 256) is False


def test_drop_last_rejects_single_sample() -> None:
    """One row cannot train BatchNorm1d."""
    with pytest.raises(ValueError, match="at least 2 training samples"):
        _drop_last_for_batchnorm(1, 256)
