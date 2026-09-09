#!/usr/bin/env python3
"""
Train TCDR from scratch on a local industrial HDF5/CSV and export an SSOI bundle.

Does not upload data. Point --data at a local path (kept outside git).

Example (paper-like Phase-1 protocol):

python examples/train_from_h5.py \\
  --data /path/to/sulfatos_dados_concatenados_formatados_2021.h5 \\
  --target 1251_FIT_801C_2 \\
  --bundle-dir artifacts/bundle_target_a
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssoi.train import load_exclude_features_file, train_from_h5  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """CLI for from-scratch industrial training."""
    p = argparse.ArgumentParser(
        description="Train SSOI/TCDR from scratch and export a deployment bundle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, help="Path to local .h5/.csv (not committed)")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument(
        "--bundle-dir",
        default="artifacts/bundle_from_scratch",
        help="Output bundle directory",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--p-mask", type=float, default=0.5)
    p.add_argument("--masked-weight", type=float, default=1.0)
    p.add_argument("--min-correlation", type=float, default=0.3)
    p.add_argument("--max-missing-pct", type=float, default=30.0)
    p.add_argument("--max-features", type=int, default=10)
    p.add_argument("--min-features", type=int, default=3)
    p.add_argument(
        "--exclude-features",
        nargs="*",
        default=None,
        help="Auxiliary tags that must not enter automatic screening",
    )
    p.add_argument(
        "--exclude-features-file",
        default=None,
        help="JSON list or object with key exclude_features",
    )
    p.add_argument("--time-test-size", type=float, default=0.05)
    p.add_argument("--time-val-size", type=float, default=0.1)
    p.add_argument("--min-target-coverage", type=float, default=0.2)
    p.add_argument("--min-target-count", type=int, default=10000)
    p.add_argument("--max-skip-head-frac", type=float, default=0.95)
    p.add_argument(
        "--hdf-key",
        default=None,
        help="HDF5 table/group name when the file contains more than one table",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA exists")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run training and print a compact JSON summary."""
    args = build_parser().parse_args(argv)
    device = "cpu" if args.cpu else None
    excluded: list[str] = []
    if args.exclude_features_file:
        excluded.extend(load_exclude_features_file(args.exclude_features_file))
    if args.exclude_features:
        excluded.extend(args.exclude_features)
    exclude_arg = excluded if excluded else None
    result = train_from_h5(
        data_path=args.data,
        target=args.target,
        bundle_dir=args.bundle_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        p_mask=args.p_mask,
        masked_weight=args.masked_weight,
        min_correlation=args.min_correlation,
        max_missing_pct=args.max_missing_pct,
        max_features=args.max_features,
        min_features=args.min_features,
        exclude_features=exclude_arg,
        test_size=args.time_test_size,
        val_size=args.time_val_size,
        min_target_coverage=args.min_target_coverage,
        min_target_count=args.min_target_count,
        max_skip_head_frac=args.max_skip_head_frac,
        hdf_key=args.hdf_key,
        device=device,
    )
    summary = {
        "bundle_dir": str(result.bundle_dir),
        "features": result.features,
        "skip_frac": result.skip_frac,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "n_test": result.n_test,
        "metrics": result.metrics,
        "paper_phase1_reference": {
            "MAE": 2.738,
            "RMSE": 4.587,
            "R2": 0.922,
            "note": "cdae_optimal_20251226_171939 Phase-1 held-out (not identical seed path)",
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
