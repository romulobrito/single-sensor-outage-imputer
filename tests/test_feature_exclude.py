#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for auxiliary exclusion in automatic feature screening.

HDF5 cases skip when the industrial file is absent. They never print
process measurements.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ssoi.bundle import load_bundle, save_bundle
from ssoi.model import TargetConditionalDenoisingRegressor
from ssoi.train import (
    load_exclude_features_file,
    normalize_exclude_features,
    select_features_train_only,
)

TARGET = "alvo"
HDF_KEY = "formatted_data_2021"
H5_ENV = "SSOI_H5_PATH"
LOCAL_H5 = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sulfatos_dados_concatenados_formatados_2021.h5"
)


def _ranking_frame(n: int = 80) -> pd.DataFrame:
    """Invented table: 'proibida' tracks the target closer than 'boa'."""
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=float)
    y = 10.0 + 0.5 * t
    return pd.DataFrame(
        {
            TARGET: y,
            "proibida": y + rng.normal(0.0, 0.01, n),
            "boa": 0.4 * y + rng.normal(0.0, 0.8, n),
            "extra": 0.35 * y + rng.normal(0.0, 0.3, n),
            "extra2": 0.3 * y + rng.normal(0.0, 0.3, n),
        }
    )


def _h5_path() -> Optional[Path]:
    """Local sulfates file via env or ignored data/ folder."""
    env = os.environ.get(H5_ENV, "")
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None
    if LOCAL_H5.is_file():
        return LOCAL_H5
    return None


def test_exclude_loses_to_automatic_ranking() -> None:
    df = _ranking_frame()
    livre = select_features_train_only(
        df, TARGET, min_features=3, max_features=4
    )
    assert livre[0] == "proibida"
    preso = select_features_train_only(
        df,
        TARGET,
        min_features=3,
        max_features=4,
        exclude_features=["proibida"],
    )
    assert "proibida" not in preso
    assert "boa" in preso


def test_empty_exclude_matches_default() -> None:
    df = _ranking_frame()
    a = select_features_train_only(df, TARGET, min_features=3, max_features=4)
    b = select_features_train_only(
        df, TARGET, min_features=3, max_features=4, exclude_features=[]
    )
    assert a == b


def test_exclude_target_is_error() -> None:
    df = _ranking_frame()
    with pytest.raises(ValueError, match="must not contain the target"):
        normalize_exclude_features(df, TARGET, [TARGET])


def test_exclude_unknown_name_is_error() -> None:
    df = _ranking_frame()
    with pytest.raises(ValueError, match="Unknown exclude_features"):
        normalize_exclude_features(df, TARGET, ["nao_existe"])


def test_exclude_file_json_list(tmp_path: Path) -> None:
    path = tmp_path / "block.json"
    path.write_text(json.dumps(["proibida", "boa"]), encoding="utf-8")
    assert load_exclude_features_file(path) == ["proibida", "boa"]


def test_exclude_file_json_object(tmp_path: Path) -> None:
    path = tmp_path / "block.json"
    path.write_text(
        json.dumps({"exclude_features": ["proibida"]}),
        encoding="utf-8",
    )
    assert load_exclude_features_file(path) == ["proibida"]


def test_save_bundle_writes_excluded_features(tmp_path: Path) -> None:
    features = ["a", "b"]
    model = TargetConditionalDenoisingRegressor(
        n_features=2, latent_dim=8, dropout=0.1
    )
    scaler_x = StandardScaler().fit([[0.0, 1.0], [1.0, 2.0]])
    scaler_y = StandardScaler().fit([[0.0], [1.0]])
    out = save_bundle(
        tmp_path / "bundle",
        features=features,
        train_feature_means={"a": 0.0, "b": 1.0},
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        model=model,
        target_name=TARGET,
        excluded_features=["proibida"],
    )
    cfg = json.loads((out / "feature_selection_config.json").read_text(encoding="utf-8"))
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert cfg["excluded_features"] == ["proibida"]
    assert man["excluded_features"] == ["proibida"]
    loaded = load_bundle(out)
    assert loaded.features == features


def test_load_bundle_without_excluded_field(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "synthetic_bundle"
    if not (src / "best_model.pth").is_file():
        pytest.skip("synthetic bundle weights are missing")
    dest = tmp_path / "old"
    import shutil

    shutil.copytree(src, dest)
    cfg_path = dest / "feature_selection_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.pop("excluded_features", None)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = load_bundle(dest)
    assert len(loaded.features) >= 1


@pytest.mark.skipif(_h5_path() is None, reason="sulfates HDF5 not available locally")
def test_h5_exclude_drops_selected_auxiliary() -> None:
    """Screening on sulfates: a banned tag must leave the automatic list."""
    pytest.importorskip("tables")
    path = _h5_path()
    assert path is not None
    df = pd.read_hdf(path, key=HDF_KEY)
    if not isinstance(df, pd.DataFrame):
        pytest.skip("HDF5 key did not yield a DataFrame")
    target = "1251_FIT_801C_2"
    if target not in df.columns:
        pytest.skip("expected sulfates target column is missing")
    # Last block is mostly observed; screening uses this slice only.
    work = df.iloc[-4000:]
    livre = select_features_train_only(
        work,
        target,
        min_features=3,
        max_features=10,
    )
    banned = livre[0]
    preso = select_features_train_only(
        work,
        target,
        min_features=3,
        max_features=10,
        exclude_features=[banned],
    )
    assert banned not in preso
    assert len(preso) >= 3


@pytest.mark.skipif(_h5_path() is None, reason="sulfates HDF5 not available locally")
def test_h5_exclude_unknown_tag_fails() -> None:
    pytest.importorskip("tables")
    path = _h5_path()
    assert path is not None
    df = pd.read_hdf(path, key=HDF_KEY)
    work = df.iloc[-200:]
    with pytest.raises(ValueError, match="Unknown exclude_features"):
        select_features_train_only(
            work,
            "1251_FIT_801C_2",
            min_features=1,
            max_features=3,
            exclude_features=["TAG_QUE_NAO_EXISTE"],
        )
