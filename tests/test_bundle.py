"""
Tests for deployment bundle loading checks.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import pytest

from ssoi.bundle import load_bundle

SRC_BUNDLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_bundle"


def _copy_bundle(tmp_path: Path) -> Path:
    """Copy the synthetic bundle into an isolated temp directory."""
    if not (SRC_BUNDLE / "best_model.pth").is_file():
        pytest.skip("synthetic bundle weights are missing")
    dest = tmp_path / "bundle"
    shutil.copytree(SRC_BUNDLE, dest)
    return dest


def test_load_bundle_missing_weights(tmp_path: Path) -> None:
    """A bundle without best_model.pth is rejected."""
    dest = _copy_bundle(tmp_path)
    (dest / "best_model.pth").unlink()
    with pytest.raises(FileNotFoundError, match="best_model.pth"):
        load_bundle(dest)


def test_load_bundle_scaler_feature_mismatch(tmp_path: Path) -> None:
    """scaler_X n_features_in_ must match the feature list."""
    dest = _copy_bundle(tmp_path)
    scaler = joblib.load(dest / "scaler_X.joblib")
    scaler.n_features_in_ = int(scaler.n_features_in_) + 1
    joblib.dump(scaler, dest / "scaler_X.joblib")
    with pytest.raises(ValueError, match="n_features_in_"):
        load_bundle(dest)


def test_load_bundle_corrupt_weights(tmp_path: Path) -> None:
    """Corrupt checkpoints mention best_model.pth in the error."""
    dest = _copy_bundle(tmp_path)
    (dest / "best_model.pth").write_bytes(b"not-a-pytorch-checkpoint")
    with pytest.raises(RuntimeError, match="best_model.pth"):
        load_bundle(dest)
