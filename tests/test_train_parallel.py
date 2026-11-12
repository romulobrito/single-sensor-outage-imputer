"""
Tests for the parallel training launcher (no real TCDR fit).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ssoi.train_jobs import (
    build_train_command,
    commands_from_config,
    load_jobs_config,
    proxima_pasta_vn,
    run_train_commands,
    validate_jobs_config,
)

_EXEMPLO = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "train_parallel.example.yaml"
)


def _cfg(**overrides: object) -> dict:
    payload: dict = {
        "data": "/tmp/fake.h5",
        "max_parallel": 2,
        "cpu": True,
        "jobs": [
            {
                "target": "TAG_VAZAO",
                "bundle_dir": "/tmp/out/VAZAO",
                "exclude_features": ["TAG_PRESSAO"],
            },
            {
                "target": "TAG_PRESSAO",
                "bundle_dir": "/tmp/out/PRESSAO",
                "exclude_features": ["TAG_VAZAO"],
            },
        ],
    }
    payload.update(overrides)
    return payload


def test_validate_rejects_same_bundle_dir() -> None:
    payload = _cfg()
    payload["jobs"][1]["bundle_dir"] = payload["jobs"][0]["bundle_dir"]
    with pytest.raises(ValueError, match="duplicate bundle_dir"):
        validate_jobs_config(payload)


def test_validate_rejects_duplicate_target() -> None:
    payload = _cfg()
    payload["jobs"][1]["target"] = "TAG_VAZAO"
    with pytest.raises(ValueError, match="duplicate target"):
        validate_jobs_config(payload)


def test_command_includes_exclude_list() -> None:
    cmd = build_train_command(
        python_exe="python",
        train_script=Path("examples/train_from_h5.py"),
        data="/tmp/d.h5",
        target="TAG_VAZAO",
        bundle_dir="/tmp/VAZAO",
        exclude_features=["TAG_PRESSAO", "TAG_RUIDO"],
        cpu=True,
    )
    assert "--target" in cmd
    assert "TAG_VAZAO" in cmd
    assert "--exclude-features" in cmd
    assert "TAG_PRESSAO" in cmd
    assert "TAG_RUIDO" in cmd


def test_commands_from_config_two_jobs() -> None:
    cmds = commands_from_config(_cfg(), python_exe="python")
    assert len(cmds) == 2
    assert cmds[0] != cmds[1]


def test_commands_from_config_forwards_epochs() -> None:
    cmds = commands_from_config(
        _cfg(epochs=1, patience=1), python_exe="python"
    )
    assert "--epochs" in cmds[0]
    assert cmds[0][cmds[0].index("--epochs") + 1] == "1"
    assert "--patience" in cmds[1]


def test_run_train_commands_runs_two_successes(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text('x')\n",
        encoding="utf-8",
    )
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    code = run_train_commands(
        [
            [sys.executable, str(script), str(a)],
            [sys.executable, str(script), str(b)],
        ],
        max_parallel=2,
    )
    assert code == 0
    assert a.is_file() and b.is_file()


def test_load_jobs_config(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(_cfg()), encoding="utf-8")
    loaded = load_jobs_config(path)
    assert loaded["jobs"][0]["target"] == "TAG_VAZAO"


def test_exemplo_yaml_abre_dois_alvos() -> None:
    loaded = load_jobs_config(_EXEMPLO)
    assert [item["target"] for item in loaded["jobs"]] == [
        "TAG_VAZAO",
        "TAG_PRESSAO",
    ]
    assert "um processo por alvo" not in loaded
    cfg = validate_jobs_config(loaded)
    assert cfg["jobs"][0]["bundle_dir"].endswith("/VAZAO/v1")
    assert cfg["jobs"][1]["bundle_dir"].endswith("/PRESSAO/v1")
    assert cfg["cpu"] is True


def test_json_antigo_ainda_carrega() -> None:
    caminho = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "train_parallel.example.json"
    )
    cfg = validate_jobs_config(load_jobs_config(caminho))
    assert len(cfg["jobs"]) == 2
    assert "--lr" not in commands_from_config(
        load_jobs_config(caminho), python_exe="python"
    )[0]


def test_chave_desconhecida_e_yaml_repetido(tmp_path: Path) -> None:
    ruim = tmp_path / "ruim.yaml"
    ruim.write_text("shadow_mode: true\ndata: /tmp/x.h5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key"):
        validate_jobs_config(load_jobs_config(ruim))
    repetido = tmp_path / "repetido.yaml"
    repetido.write_text("epochs: 1\nepochs: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_jobs_config(repetido)


def test_lr_escrito_e_cpu(tmp_path: Path) -> None:
    raiz_a = tmp_path / "VAZAO"
    raiz_b = tmp_path / "PRESSAO"
    payload = _cfg(lr=0.001, cpu=False)
    payload["jobs"][0]["bundle_dir"] = str(raiz_a)
    payload["jobs"][1]["bundle_dir"] = str(raiz_b)
    cmds = commands_from_config(payload, python_exe="python")
    assert cmds[0][cmds[0].index("--lr") + 1] == "0.001"
    assert "--cpu" not in cmds[0]
    assert cmds[0][cmds[0].index("--bundle-dir") + 1].endswith("/VAZAO/v1")
    forcado = commands_from_config(_cfg(cpu=True), python_exe="python")
    assert "--cpu" in forcado[0]


def test_proxima_pasta_pula_v1_com_manifesto(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / "manifest.json").write_text("{}", encoding="utf-8")
    assert proxima_pasta_vn(tmp_path) == tmp_path / "v2"
    assert proxima_pasta_vn(tmp_path / "vazia") == tmp_path / "vazia" / "v1"
