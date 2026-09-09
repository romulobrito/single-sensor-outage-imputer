#!/usr/bin/env python3
"""
Run one-target SSOI trainings in parallel (one process per target).

The training code is unchanged: this script only starts N copies of
``train_from_h5.py``. Each job needs its own --bundle-dir.

Example::

    python examples/train_parallel.py --config jobs.json

The JSON is not committed with industrial paths. Shape::

    {
      "data": "/path/to/file.h5",
      "hdf_key": "formatted_data_2021",
      "max_parallel": 2,
      "epochs": 100,
      "cpu": true,
      "jobs": [
        {
          "target": "TAG_VAZAO",
          "bundle_dir": "artifacts/VAZAO",
          "exclude_features": ["TAG_PRESSAO"]
        },
        {
          "target": "TAG_PRESSAO",
          "bundle_dir": "artifacts/PRESSAO",
          "exclude_features": ["TAG_VAZAO"]
        }
      ]
    }
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssoi.train_jobs import (  # noqa: E402
    commands_from_config,
    load_jobs_config,
    run_train_commands,
    validate_jobs_config,
)


def main(argv: list[str] | None = None) -> int:
    """Load the job list and start trainings, capped by max_parallel."""
    parser = argparse.ArgumentParser(
        description="Launch several SSOI trainings in parallel"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON with data path and jobs (not an industrial dump)",
    )
    args = parser.parse_args(argv)
    payload = load_jobs_config(args.config)
    cfg = validate_jobs_config(payload)
    commands = commands_from_config(payload)
    print(f"Starting {len(commands)} trainings, max_parallel={cfg['max_parallel']}")
    for job in cfg["jobs"]:
        extra = ",".join(job["exclude_features"]) or "(none)"
        print(f"  {job['target']} -> {job['bundle_dir']} exclude={extra}")
    return run_train_commands(
        commands, max_parallel=cfg["max_parallel"], cwd=ROOT
    )


if __name__ == "__main__":
    raise SystemExit(main())
