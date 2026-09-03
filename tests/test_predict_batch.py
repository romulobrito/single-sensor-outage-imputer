"""
Tests for batched VirtualSensor.predict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssoi import VirtualSensor

BUNDLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "synthetic_bundle"


@pytest.fixture(scope="module")
def sensor() -> VirtualSensor:
    """Load the checked-in synthetic smoke bundle."""
    if not (BUNDLE_DIR / "best_model.pth").is_file():
        pytest.skip("synthetic bundle weights are missing")
    return VirtualSensor.from_bundle(BUNDLE_DIR, device="cpu")


def test_predict_chunks_match_full_forward(sensor: VirtualSensor) -> None:
    """Chunked inference must match a single larger batch."""
    n_features = len(sensor.features)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, n_features))
    full = sensor.predict(x, batch_size=1024)
    chunked = sensor.predict(x, batch_size=2)
    np.testing.assert_allclose(chunked, full, rtol=1e-5, atol=1e-5)
    assert full.shape == (5,)


def test_predict_single_row(sensor: VirtualSensor) -> None:
    """Eval-mode BatchNorm accepts a single sample."""
    n_features = len(sensor.features)
    x = np.zeros((1, n_features), dtype=np.float64)
    pred = sensor.predict(x, batch_size=1)
    assert pred.shape == (1,)
    assert np.isfinite(pred).all()


def test_predict_rejects_invalid_batch_size(sensor: VirtualSensor) -> None:
    """batch_size must be a positive integer."""
    n_features = len(sensor.features)
    x = np.zeros((2, n_features), dtype=np.float64)
    with pytest.raises(ValueError, match="batch_size"):
        sensor.predict(x, batch_size=0)
