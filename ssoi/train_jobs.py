#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch several one-target trainings as separate processes.

Each job is still one target, one exclude list, one output folder.
This module only starts those processes at the same time (capped).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

PathLike = Union[str, Path]

# Optional shared CLI flags forwarded to every train_from_h5.py process.
_INT_FLAGS = {
    "epochs": "--epochs",
    "patience": "--patience",
    "seed": "--seed",
    "batch_size": "--batch-size",
    "max_features": "--max-features",
    "min_features": "--min-features",
    "latent_dim": "--latent-dim",
    "min_target_count": "--min-target-count",
}
_FLOAT_FLAGS = {
    "lr": "--lr",
    "dropout": "--dropout",
    "p_mask": "--p-mask",
    "masked_weight": "--masked-weight",
    "min_correlation": "--min-correlation",
    "max_missing_pct": "--max-missing-pct",
    "time_test_size": "--time-test-size",
    "time_val_size": "--time-val-size",
    "min_target_coverage": "--min-target-coverage",
    "max_skip_head_frac": "--max-skip-head-frac",
}
_CHAVES_TOPO = frozenset(
    {
        "data",
        "hdf_key",
        "max_parallel",
        "cpu",
        "jobs",
        *_INT_FLAGS,
        *_FLOAT_FLAGS,
    }
)
_CHAVES_ALVO = frozenset({"target", "bundle_dir", "exclude_features"})


class _LoaderSemChaveRepetida(yaml.SafeLoader):
    """SafeLoader that rejects a key written twice in one block."""


def _mapa_sem_repetir(loader: yaml.SafeLoader, node: yaml.Node) -> dict:
    """Build a dict and stop if the key already appeared."""
    loader.flatten_mapping(node)
    vistos = set()
    for key_node, _value_node in node.value:
        chave = loader.construct_object(key_node, deep=False)
        if chave in vistos:
            raise ValueError(f"duplicate YAML key {chave!r}")
        vistos.add(chave)
    mapeado = yaml.SafeLoader.construct_mapping(loader, node, deep=True)
    if not isinstance(mapeado, dict):
        raise ValueError("YAML mapping did not build a dict")
    return mapeado


_LoaderSemChaveRepetida.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _mapa_sem_repetir,
)


def _inteiro(nome: str, val: Any, *, minimo: int) -> int:
    """Accept a real int. A bool is an int in Python and is refused."""
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{nome} must be an int")
    if val < minimo:
        raise ValueError(f"{nome} must be >= {minimo}")
    return val


def _decimal(nome: str, val: Any) -> float:
    """Accept an int or a float. A bool is refused."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{nome} must be a number")
    return float(val)


def _chaves_desconhecidas(
    payload: Mapping[str, Any],
    permitidas: frozenset,
    nome: str,
) -> None:
    """Stop when the file has a key this reader does not use."""
    extras = [chave for chave in payload if chave not in permitidas]
    if extras:
        lista = ", ".join(sorted(_CHAVES_TOPO if nome == "config" else _CHAVES_ALVO))
        raise ValueError(f"unknown {nome} key {extras[0]!r}; allowed: {lista}")


def proxima_pasta_vn(raiz: Path) -> Path:
    """
    Return the next free ``vN`` folder under ``raiz``.

    A version is taken when ``manifest.json`` is already there. The function
    does not create the folder and does not delete an older version.
    """
    numero = 1
    while (raiz / f"v{numero}" / "manifest.json").is_file():
        numero += 1
    return raiz / f"v{numero}"


def load_jobs_config(path: PathLike) -> Dict[str, Any]:
    """Read the JSON or YAML file that lists data path and per-target jobs."""
    raw = Path(path)
    if not raw.is_file():
        raise FileNotFoundError(f"jobs config not found: {raw}")
    texto = raw.read_text(encoding="utf-8")
    sufixo = raw.suffix.lower()
    if sufixo in {".yaml", ".yml"}:
        payload = yaml.load(texto, Loader=_LoaderSemChaveRepetida)
    elif sufixo == ".json":
        payload = json.loads(texto)
    else:
        raise ValueError("jobs config must be a .json, .yaml, or .yml file")
    if not isinstance(payload, dict):
        raise ValueError("jobs config must be a mapping")
    return payload


def _flags_extras(payload: Mapping[str, Any]) -> Dict[str, float]:
    """Read optional numeric flags. Missing keys stay at the script default."""
    extra: Dict[str, float] = {}
    for key in _INT_FLAGS:
        if key not in payload:
            continue
        minimo = 0 if key == "seed" else 1
        extra[key] = _inteiro(f"config.{key}", payload[key], minimo=minimo)
    for key in _FLOAT_FLAGS:
        if key not in payload:
            continue
        extra[key] = _decimal(f"config.{key}", payload[key])
    return extra


def validate_jobs_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Check required fields and that targets and output roots do not clash.

    Each ``bundle_dir`` is a root. The returned path is the next free ``vN``
    under that root. Returns a normalized copy.
    """
    _chaves_desconhecidas(payload, _CHAVES_TOPO, "config")
    data = payload.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("config.data must be a non-empty string")
    jobs_raw = payload.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("config.jobs must be a non-empty list")
    max_parallel = _inteiro(
        "config.max_parallel",
        payload.get("max_parallel", 2),
        minimo=1,
    )
    if "cpu" in payload and not isinstance(payload.get("cpu"), bool):
        raise ValueError("config.cpu must be a bool")
    cpu = True if "cpu" not in payload else bool(payload.get("cpu"))

    jobs: List[Dict[str, Any]] = []
    targets = []
    dirs = []
    for i, item in enumerate(jobs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"jobs[{i}] must be an object")
        _chaves_desconhecidas(item, _CHAVES_ALVO, "job")
        target = item.get("target")
        bundle_dir = item.get("bundle_dir")
        if not isinstance(target, str) or not target:
            raise ValueError(f"jobs[{i}].target must be a non-empty string")
        if not isinstance(bundle_dir, str) or not bundle_dir:
            raise ValueError(f"jobs[{i}].bundle_dir must be a non-empty string")
        excluded = item.get("exclude_features", [])
        if excluded is None:
            excluded = []
        if not isinstance(excluded, list) or not all(
            isinstance(name, str) and name for name in excluded
        ):
            raise ValueError(
                f"jobs[{i}].exclude_features must be a list of non-empty strings"
            )
        raiz = Path(bundle_dir).expanduser().resolve()
        pasta = proxima_pasta_vn(raiz)
        jobs.append(
            {
                "target": target,
                "bundle_dir": str(pasta),
                "exclude_features": list(excluded),
            }
        )
        targets.append(target)
        dirs.append(str(raiz))

    if len(set(targets)) != len(targets):
        raise ValueError("duplicate target in config.jobs")
    if len(set(dirs)) != len(dirs):
        raise ValueError("duplicate bundle_dir in config.jobs")

    hdf_key = payload.get("hdf_key")
    if hdf_key is not None and not isinstance(hdf_key, str):
        raise ValueError("config.hdf_key must be a string when set")

    return {
        "data": data,
        "hdf_key": hdf_key,
        "cpu": cpu,
        "max_parallel": max_parallel,
        "jobs": jobs,
        "extra_flags": _flags_extras(payload),
    }


def _argv_value(argv: Sequence[str], flag: str) -> str:
    """Return the argument after ``flag``, or a placeholder."""
    listed = list(argv)
    if flag not in listed:
        return "?"
    idx = listed.index(flag)
    if idx + 1 >= len(listed):
        return "?"
    return str(listed[idx + 1])


def build_train_command(
    *,
    python_exe: str,
    train_script: Path,
    data: str,
    target: str,
    bundle_dir: str,
    exclude_features: Sequence[str],
    hdf_key: Optional[str] = None,
    cpu: bool = True,
    extra_flags: Optional[Mapping[str, float]] = None,
) -> List[str]:
    """Build the argv for one ``train_from_h5.py`` process."""
    cmd = [
        python_exe,
        str(train_script),
        "--data",
        data,
        "--target",
        target,
        "--bundle-dir",
        bundle_dir,
    ]
    if hdf_key:
        cmd.extend(["--hdf-key", hdf_key])
    if cpu:
        cmd.append("--cpu")
    extras = extra_flags or {}
    for key, flag in {**_INT_FLAGS, **_FLOAT_FLAGS}.items():
        if key in extras:
            cmd.extend([flag, str(extras[key])])
    if exclude_features:
        cmd.append("--exclude-features")
        cmd.extend(list(exclude_features))
    return cmd


def run_train_commands(
    commands: Sequence[Sequence[str]],
    *,
    max_parallel: int,
    cwd: Optional[PathLike] = None,
) -> int:
    """
    Run commands concurrently, at most ``max_parallel`` at a time.

    Returns 0 if every process exits 0; otherwise the first non-zero code.
    """
    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    if not commands:
        raise ValueError("commands must be a non-empty sequence")

    workdir = str(cwd) if cwd is not None else None
    codes: List[int] = []

    def _one(argv: Sequence[str]) -> int:
        target = _argv_value(argv, "--target")
        print(f"[parallel] start target={target}", flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(list(argv), cwd=workdir, check=False, env=env)
        print(
            f"[parallel] end target={target} code={int(proc.returncode)}",
            flush=True,
        )
        return int(proc.returncode)

    with ThreadPoolExecutor(max_workers=int(max_parallel)) as pool:
        futures = [pool.submit(_one, cmd) for cmd in commands]
        for fut in as_completed(futures):
            codes.append(int(fut.result()))
    failed = [c for c in codes if c != 0]
    return failed[0] if failed else 0


def commands_from_config(
    payload: Mapping[str, Any],
    *,
    python_exe: Optional[str] = None,
    train_script: Optional[PathLike] = None,
) -> List[List[str]]:
    """Turn a validated config into argv lists."""
    cfg = validate_jobs_config(payload)
    exe = python_exe if python_exe else sys.executable
    script = (
        Path(train_script)
        if train_script is not None
        else Path(__file__).resolve().parents[1] / "examples" / "train_from_h5.py"
    )
    cmds: List[List[str]] = []
    for job in cfg["jobs"]:
        cmds.append(
            build_train_command(
                python_exe=exe,
                train_script=script,
                data=cfg["data"],
                target=job["target"],
                bundle_dir=job["bundle_dir"],
                exclude_features=job["exclude_features"],
                hdf_key=cfg["hdf_key"],
                cpu=cfg["cpu"],
                extra_flags=cfg["extra_flags"],
            )
        )
    return cmds
