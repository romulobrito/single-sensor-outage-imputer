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

PathLike = Union[str, Path]

# Optional shared CLI flags forwarded to every train_from_h5.py process.
_INT_FLAGS = {
    "epochs": "--epochs",
    "patience": "--patience",
    "seed": "--seed",
    "batch_size": "--batch-size",
    "max_features": "--max-features",
    "min_features": "--min-features",
}


def _optional_int_flags(payload: Mapping[str, Any]) -> Dict[str, int]:
    """Read optional integer train flags shared by all jobs."""
    extra: Dict[str, int] = {}
    for key in _INT_FLAGS:
        if key not in payload:
            continue
        val = payload[key]
        if not isinstance(val, int):
            raise ValueError(f"config.{key} must be an int")
        if key != "seed" and val < 1:
            raise ValueError(f"config.{key} must be >= 1")
        extra[key] = val
    return extra


def load_jobs_config(path: PathLike) -> Dict[str, Any]:
    """Read the JSON file that lists data path and per-target jobs."""
    raw = Path(path)
    if not raw.is_file():
        raise FileNotFoundError(f"jobs config not found: {raw}")
    payload = json.loads(raw.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("jobs config must be a JSON object")
    return payload


def validate_jobs_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Check required fields and that targets and output folders do not clash.

    Returns a normalized copy (jobs as list of dicts with string paths).
    """
    data = payload.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("config.data must be a non-empty string")
    jobs_raw = payload.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("config.jobs must be a non-empty list")
    max_parallel = payload.get("max_parallel", 2)
    if not isinstance(max_parallel, int) or max_parallel < 1:
        raise ValueError("config.max_parallel must be an int >= 1")

    jobs: List[Dict[str, Any]] = []
    targets = []
    dirs = []
    for i, item in enumerate(jobs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"jobs[{i}] must be an object")
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
        jobs.append(
            {
                "target": target,
                "bundle_dir": bundle_dir,
                "exclude_features": list(excluded),
            }
        )
        targets.append(target)
        dirs.append(str(Path(bundle_dir).expanduser().resolve()))

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
        "cpu": bool(payload.get("cpu", True)),
        "max_parallel": max_parallel,
        "jobs": jobs,
        "extra_flags": _optional_int_flags(payload),
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
    extra_flags: Optional[Mapping[str, int]] = None,
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
    for key, flag in _INT_FLAGS.items():
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
