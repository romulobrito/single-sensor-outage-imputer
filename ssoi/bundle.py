"""
Bundle I/O for deployment artifacts.

A bundle is a directory containing:
  - manifest.json
  - feature_selection_config.json  (feature column order)
  - train_feature_means.json       (per-feature fill values)
  - scaler_X.joblib / scaler_y.joblib
  - model_config.json
  - best_model.pth
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from ssoi.model import TargetConditionalDenoisingRegressor

PathLike = Union[str, Path]


@dataclass
class ImputerBundle:
    """In-memory deployment bundle for single-sensor outage imputation."""

    features: List[str]
    train_feature_means: Dict[str, float]
    scaler_x: Any
    scaler_y: Any
    model: TargetConditionalDenoisingRegressor
    target_name: str
    manifest: Dict[str, Any]
    model_config: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")


def _extract_feature_list(feature_cfg: Mapping[str, Any]) -> List[str]:
    """Accept common feature-config layouts used in the research pipeline."""
    if "selected_features" in feature_cfg:
        feats = feature_cfg["selected_features"]
    elif "features" in feature_cfg:
        feats = feature_cfg["features"]
    else:
        raise KeyError(
            "feature_selection_config.json must contain "
            "'selected_features' or 'features'"
        )
    if not isinstance(feats, list) or not feats:
        raise ValueError("Feature list must be a non-empty list of strings")
    out: List[str] = []
    for item in feats:
        if not isinstance(item, str) or not item:
            raise ValueError(f"Invalid feature name: {item!r}")
        out.append(item)
    return out


def _means_to_dict(
    features: List[str],
    means_raw: Mapping[str, Any],
) -> Dict[str, float]:
    """Normalize means JSON to feature -> float."""
    # Allow {"means": {...}} or flat {feature: value}
    if "means" in means_raw and isinstance(means_raw["means"], dict):
        source = means_raw["means"]
    else:
        source = means_raw
    out: Dict[str, float] = {}
    for name in features:
        if name not in source:
            raise KeyError(f"Missing train mean for feature '{name}'")
        out[name] = float(source[name])
    return out


def save_bundle(
    bundle_dir: PathLike,
    *,
    features: List[str],
    train_feature_means: Mapping[str, float],
    scaler_x: Any,
    scaler_y: Any,
    model: TargetConditionalDenoisingRegressor,
    target_name: str,
    manifest: Optional[Mapping[str, Any]] = None,
    excluded_features: Optional[Sequence[str]] = None,
) -> Path:
    """
    Persist a trainable/deployable bundle to disk.

    Returns the bundle directory path.
    """
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)

    excluded = [str(name) for name in (excluded_features or []) if str(name)]
    feature_cfg = {
        "selected_features": list(features),
        "target": target_name,
        "excluded_features": excluded,
    }
    means_payload = {"means": {k: float(v) for k, v in train_feature_means.items()}}
    model_cfg = model.config_dict()
    man = {
        "schema_version": "1.0",
        "package": "single-sensor-outage-imputer",
        "target_name": target_name,
        "n_features": len(features),
        "inference_mode": "missing_target",
        "inference_channels": {"t_tilde": 0.0, "m_t": 1.0},
        "excluded_features": excluded,
    }
    if manifest:
        man.update(dict(manifest))
    man["excluded_features"] = excluded

    _write_json(root / "feature_selection_config.json", feature_cfg)
    _write_json(root / "train_feature_means.json", means_payload)
    _write_json(root / "model_config.json", model_cfg)
    _write_json(root / "manifest.json", man)
    joblib.dump(scaler_x, root / "scaler_X.joblib")
    joblib.dump(scaler_y, root / "scaler_y.joblib")
    torch.save(model.state_dict(), root / "best_model.pth")
    return root


def load_bundle(bundle_dir: PathLike, device: str = "cpu") -> ImputerBundle:
    """
    Load a deployment bundle from disk.

    Args:
        bundle_dir: directory with manifest, scalers, and weights
        device: torch device string for the loaded model
    """
    root = Path(bundle_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {root}")

    required = [
        "manifest.json",
        "feature_selection_config.json",
        "train_feature_means.json",
        "model_config.json",
        "scaler_X.joblib",
        "scaler_y.joblib",
        "best_model.pth",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle incomplete at {root}: missing {missing}")

    manifest = _read_json(root / "manifest.json")
    feature_cfg = _read_json(root / "feature_selection_config.json")
    means_raw = _read_json(root / "train_feature_means.json")
    model_cfg = _read_json(root / "model_config.json")

    features = _extract_feature_list(feature_cfg)
    means = _means_to_dict(features, means_raw)
    target_name = str(
        manifest.get("target_name")
        or feature_cfg.get("target")
        or "target"
    )

    n_features = int(model_cfg.get("n_features", len(features)))
    if n_features != len(features):
        raise ValueError(
            f"model_config n_features={n_features} != len(features)={len(features)}"
        )
    latent_dim = int(model_cfg.get("latent_dim", 16))
    dropout = float(model_cfg.get("dropout", 0.2))

    model = TargetConditionalDenoisingRegressor(
        n_features=n_features,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    weights_path = root / "best_model.pth"
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model weights from {weights_path}: {exc}"
        ) from exc
    model.to(device)
    model.eval()

    scaler_x = joblib.load(root / "scaler_X.joblib")
    scaler_y = joblib.load(root / "scaler_y.joblib")
    if not isinstance(scaler_x, StandardScaler):
        # Accept RobustScaler or other sklearn transformers with transform()
        if not hasattr(scaler_x, "transform"):
            raise TypeError("scaler_X must expose .transform()")
    n_in = getattr(scaler_x, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(features):
        raise ValueError(
            f"scaler_X n_features_in_={int(n_in)} != len(features)={len(features)}"
        )
    if not hasattr(scaler_y, "inverse_transform"):
        raise TypeError("scaler_y must expose .inverse_transform()")

    return ImputerBundle(
        features=features,
        train_feature_means=means,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        model=model,
        target_name=target_name,
        manifest=manifest,
        model_config=model_cfg,
    )
