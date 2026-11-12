"""
Occupied bundle folders are refused before any fit. No HDF5.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ssoi.train import recusar_pasta_ocupada


def _parser():
    caminho = (
        Path(__file__).resolve().parents[1] / "examples" / "train_from_h5.py"
    )
    spec = importlib.util.spec_from_file_location("train_from_h5_cli", caminho)
    if spec is None or spec.loader is None:
        raise AssertionError("train script did not load")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.build_parser()


def test_pasta_com_manifesto_recusa(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        recusar_pasta_ocupada(tmp_path)


def test_overwrite_e_pasta_vazia_passam(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    recusar_pasta_ocupada(tmp_path, overwrite=True)
    recusar_pasta_ocupada(tmp_path / "vazia")
    assert (tmp_path / "manifest.json").is_file()


def test_flag_overwrite_so_no_script() -> None:
    parser = _parser()
    sem = parser.parse_args(["--data", "d.h5", "--target", "TAG"])
    com = parser.parse_args(
        ["--data", "d.h5", "--target", "TAG", "--overwrite"]
    )
    assert sem.overwrite is False
    assert com.overwrite is True
