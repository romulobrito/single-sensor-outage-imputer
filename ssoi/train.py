"""
Leakage-safe training utilities for TCDR on industrial tables.

Implements blocked chronological split with adaptive head-skip, train-only
feature screening, scaling, mask-weighted training, and bundle export.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ssoi.bundle import save_bundle
from ssoi.model import TargetConditionalDenoisingRegressor
from ssoi.predict import VirtualSensor

PathLike = Union[str, Path]


@dataclass
class TrainResult:
    """Artifacts and metrics produced by a from-scratch training run."""

    bundle_dir: Path
    features: List[str]
    metrics: Dict[str, float]
    skip_frac: float
    n_train: int
    n_val: int
    n_test: int
    history: Dict[str, List[float]]


def set_seed(seed: int) -> None:
    """Seed numpy and torch for reproducibility."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_table(data_path: PathLike, hdf_key: Optional[str] = None) -> pd.DataFrame:
    """
    Load HDF5 or CSV into a DataFrame.

    Parameters
    ----------
    data_path :
        Path to a ``.csv``, ``.h5``, or ``.hdf5`` file.
    hdf_key :
        Optional HDF5 table/group name when the file contains more than one
        dataset. Ignored for CSV.

    Returns
    -------
    pd.DataFrame
        Loaded table.

    Raises
    ------
    ImportError
        If HDF5 is requested and PyTables is not installed.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            if hdf_key is None:
                return pd.read_hdf(path)
            return pd.read_hdf(path, key=hdf_key)
        except ImportError as exc:
            raise ImportError(
                "HDF5 loading requires PyTables. Install with: "
                'pip install -e ".[hdf5]"'
            ) from exc
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    """
    Coerce a column to float, mapping invalid tokens and infinities to NaN.

    Parameters
    ----------
    series :
        Raw column, possibly object/string typed.

    Returns
    -------
    pd.Series
        Float series aligned to the input index.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def coerce_numeric_frame(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return a copy with selected columns coerced to numeric.

    Parameters
    ----------
    df :
        Input table. The caller DataFrame is not mutated.
    columns :
        Columns to coerce. When None, every column is coerced.

    Returns
    -------
    pd.DataFrame
        Copy with coerced columns.
    """
    out = df.copy()
    cols = list(out.columns) if columns is None else list(columns)
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not in DataFrame")
        out[col] = coerce_numeric_series(out[col])
    return out


def _outage_channels(yb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return production outage channels matching VirtualSensor inference.

    Parameters
    ----------
    yb :
        Target batch with shape (batch, 1).

    Returns
    -------
    tuple of torch.Tensor
        ``(t_tilde, m_t)`` with zeros and ones, same shape as ``yb``.
    """
    m_t = torch.ones_like(yb)
    t_tilde = torch.zeros_like(yb)
    return t_tilde, m_t


def blocked_split_with_skip(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.05,
    val_size: float = 0.1,
    min_target_coverage: float = 0.2,
    min_target_count: int = 10000,
    skip_head_frac: float = 0.0,
    auto_skip_until_coverage: bool = True,
    max_skip_head_frac: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Contiguous blocked split using row position as time, with optional head-skip.

    Coverage counts only finite numeric target values. Invalid strings and
    infinities are treated as missing. Returned arrays are integer positions
    suitable for ``DataFrame.iloc``, not index labels.

    When ``auto_skip_until_coverage`` is True and no window meets both
    ``min_target_coverage`` and ``min_target_count``, a ``ValueError`` is
    raised with the best coverage and count observed.
    """
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in columns")
    n_total = int(len(df))
    if n_total < 3:
        raise ValueError("Dataset too small for blocked split")
    positions = np.arange(n_total, dtype=np.int64)
    present = coerce_numeric_series(df[target]).notna().to_numpy()

    def compute_from(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(len(arr))
        n_test = max(1, int(np.ceil(n * float(test_size))))
        n_rem = max(1, n - n_test)
        n_val = max(1, int(np.ceil(n_rem * float(val_size))))
        if (n_test + n_val) >= n:
            raise ValueError("Blocked split sizes too large for dataset length")
        test_pos = arr[-n_test:]
        val_pos = arr[-(n_test + n_val):-n_test]
        train_pos = arr[: (n - n_test - n_val)]
        return train_pos, val_pos, test_pos

    start_frac = min(max(0.0, float(skip_head_frac)), 0.95)
    min_cov = float(min_target_coverage)
    min_count = int(min_target_count)

    if auto_skip_until_coverage:
        upper = float(min(max(float(max_skip_head_frac), start_frac), 0.95))
        best_coverage = -1.0
        best_present = -1
        best_frac = float(start_frac)
        for frac in np.linspace(start_frac, upper, num=51):
            start_i = int(np.floor(n_total * float(frac)))
            arr = positions[start_i:]
            if len(arr) < 10:
                continue
            try:
                train_pos, val_pos, test_pos = compute_from(arr)
            except ValueError:
                continue
            total_train = int(len(train_pos))
            present_train = int(present[train_pos].sum())
            coverage = float(present_train) / float(total_train) if total_train else 0.0
            if coverage > best_coverage or (
                coverage == best_coverage and present_train > best_present
            ):
                best_coverage = coverage
                best_present = present_train
                best_frac = float(frac)
            if coverage >= min_cov and present_train >= min_count:
                return train_pos, val_pos, test_pos, float(frac)
        raise ValueError(
            "Adaptive head-skip found no window meeting "
            f"min_target_coverage={min_cov} and min_target_count={min_count}. "
            f"Best coverage={best_coverage:.4f}, "
            f"best observed count={best_present}, "
            f"max skip_frac tested={best_frac:.4f}."
        )

    start_i = int(np.floor(n_total * start_frac))
    train_pos, val_pos, test_pos = compute_from(positions[start_i:])
    return train_pos, val_pos, test_pos, float(start_frac)


def select_features_train_only(
    df_train: pd.DataFrame,
    target: str,
    min_correlation: float = 0.3,
    max_missing_pct: float = 30.0,
    min_features: int = 3,
    max_features: int = 10,
) -> List[str]:
    """
    Screen numeric auxiliaries on the training fold only.

    Ranking score = |corr| * completeness_fraction.
    """
    if target not in df_train.columns:
        raise KeyError(f"Target '{target}' not in training columns")
    y = coerce_numeric_series(df_train[target])
    candidates: List[Tuple[str, float, float, float]] = []
    for col in df_train.columns:
        if col == target:
            continue
        series = coerce_numeric_series(df_train[col])
        if series.notna().sum() < 10:
            continue
        missing_pct = float(series.isna().mean() * 100.0)
        if missing_pct > float(max_missing_pct):
            continue
        # Align observed pairs
        mask = y.notna() & series.notna()
        if int(mask.sum()) < 10:
            continue
        corr = float(series[mask].corr(y[mask]))
        if not np.isfinite(corr):
            continue
        if abs(corr) < float(min_correlation):
            continue
        completeness = 1.0 - (missing_pct / 100.0)
        score = abs(corr) * completeness
        candidates.append((col, abs(corr), completeness, score))

    candidates.sort(key=lambda t: t[3], reverse=True)
    selected = [c[0] for c in candidates[: int(max_features)]]
    if len(selected) < int(min_features):
        raise ValueError(
            f"Only {len(selected)} features passed screening; "
            f"need at least {min_features}. Relax thresholds."
        )
    return selected


def _train_model(
    model: TargetConditionalDenoisingRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    p_mask: float,
    masked_weight: float,
    device: str,
) -> Dict[str, List[float]]:
    """
    Mask-weighted MSE training with early stopping on outage validation loss.

    Training still samples random target masks. Checkpoint selection, the
    plateau scheduler, and ``history["val_loss"]`` use production channels
    ``m_t=1`` and ``t_tilde=0``.
    """
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train.astype(np.float32)),
            torch.from_numpy(y_train.reshape(-1, 1).astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val.astype(np.float32)),
            torch.from_numpy(y_val.reshape(-1, 1).astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    model.to(device)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stall = 0

    for epoch in range(int(epochs)):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            m_t = (torch.rand_like(yb) < float(p_mask)).float()
            t_tilde = (1.0 - m_t) * yb
            optimizer.zero_grad()
            y_hat = model(xb, t_tilde, m_t)
            per = criterion(y_hat, yb)
            loss = ((1.0 + float(masked_weight) * m_t) * per).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss /= max(1, n_batches)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                t_tilde, m_t = _outage_channels(yb)
                y_hat = model(xb, t_tilde, m_t)
                per = criterion(y_hat, yb)
                loss = ((1.0 + float(masked_weight) * m_t) * per).mean()
                val_loss += float(loss.item())
                n_val += 1
        val_loss /= max(1, n_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train={train_loss:.6f} | val_outage={val_loss:.6f}"
            )
        if stall >= int(patience):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


def train_from_dataframe(
    df: pd.DataFrame,
    target: str,
    bundle_dir: PathLike,
    *,
    seed: int = 42,
    min_correlation: float = 0.3,
    max_missing_pct: float = 30.0,
    min_features: int = 3,
    max_features: int = 10,
    test_size: float = 0.05,
    val_size: float = 0.1,
    min_target_coverage: float = 0.2,
    min_target_count: int = 10000,
    max_skip_head_frac: float = 0.95,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 20,
    latent_dim: int = 16,
    dropout: float = 0.2,
    p_mask: float = 0.5,
    masked_weight: float = 1.0,
    device: Optional[str] = None,
) -> TrainResult:
    """
    Full leakage-safe train pipeline ending in a deployable SSOI bundle.
    """
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in columns")

    # Numeric cleanup before split, coverage, and scaling.
    work = coerce_numeric_frame(df, list(df.columns))
    train_pos, val_pos, test_pos, skip_frac = blocked_split_with_skip(
        work,
        target=target,
        test_size=test_size,
        val_size=val_size,
        min_target_coverage=min_target_coverage,
        min_target_count=min_target_count,
        auto_skip_until_coverage=True,
        max_skip_head_frac=max_skip_head_frac,
    )
    print(f"Head skip fraction used: {skip_frac:.3f}")

    features = select_features_train_only(
        work.iloc[train_pos],
        target=target,
        min_correlation=min_correlation,
        max_missing_pct=max_missing_pct,
        min_features=min_features,
        max_features=max_features,
    )
    print(f"Selected {len(features)} features: {features}")

    observed = work[target].notna().to_numpy()

    def keep_observed(pos: np.ndarray) -> np.ndarray:
        """Keep split positions whose target is finite after numeric coerce."""
        return np.asarray(pos[observed[pos]], dtype=np.int64)

    train_i = keep_observed(train_pos)
    val_i = keep_observed(val_pos)
    test_i = keep_observed(test_pos)
    if min(len(train_i), len(val_i), len(test_i)) == 0:
        raise ValueError("Empty train/val/test after filtering to observed targets")

    x_all = work[features]
    y_all = work[[target]]
    means_series = x_all.iloc[train_i].mean()
    if means_series.isna().any():
        bad = means_series[means_series.isna()].index.tolist()
        raise ValueError(f"Cannot compute train means for: {bad}")
    means = {str(k): float(v) for k, v in means_series.items()}

    def impute(part: pd.DataFrame) -> np.ndarray:
        return part.fillna(means_series).fillna(0.0).to_numpy(dtype=np.float32)

    x_train_raw = impute(x_all.iloc[train_i])
    x_val_raw = impute(x_all.iloc[val_i])
    x_test_raw = impute(x_all.iloc[test_i])
    y_train_raw = y_all.iloc[train_i].to_numpy(dtype=np.float32)
    y_val_raw = y_all.iloc[val_i].to_numpy(dtype=np.float32)
    y_test_raw = y_all.iloc[test_i].to_numpy(dtype=np.float32)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train = scaler_x.fit_transform(x_train_raw).astype(np.float32)
    x_val = scaler_x.transform(x_val_raw).astype(np.float32)
    x_test = scaler_x.transform(x_test_raw).astype(np.float32)
    y_train = scaler_y.fit_transform(y_train_raw).astype(np.float32).ravel()
    y_val = scaler_y.transform(y_val_raw).astype(np.float32).ravel()

    model = TargetConditionalDenoisingRegressor(
        n_features=len(features),
        latent_dim=latent_dim,
        dropout=dropout,
    )
    history = _train_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        p_mask=p_mask,
        masked_weight=masked_weight,
        device=device,
    )

    out = Path(bundle_dir)
    save_bundle(
        out,
        features=features,
        train_feature_means=means,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        model=model,
        target_name=target,
        manifest={
            "seed": int(seed),
            "skip_frac": float(skip_frac),
            "p_mask": float(p_mask),
            "masked_weight": float(masked_weight),
            "n_train": int(len(train_i)),
            "n_val": int(len(val_i)),
            "n_test": int(len(test_i)),
            "train_mode": "from_scratch_blocked_split",
        },
    )

    # Evaluate with the public inference API (outage mode)
    sensor = VirtualSensor.from_bundle(out, device=device)
    y_pred = sensor.predict(x_test_raw)
    y_true = y_test_raw.ravel()
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_test": float(len(y_true)),
    }
    print(
        f"Test metrics (outage inference): "
        f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} n={len(y_true)}"
    )

    metrics_path = out / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "metrics": metrics,
                "features": features,
                "skip_frac": skip_frac,
                "history_tail": {
                    "train_loss": history["train_loss"][-5:],
                    "val_loss": history["val_loss"][-5:],
                },
            },
            fh,
            indent=2,
        )
        fh.write("\n")

    return TrainResult(
        bundle_dir=out,
        features=features,
        metrics=metrics,
        skip_frac=float(skip_frac),
        n_train=len(train_i),
        n_val=len(val_i),
        n_test=len(test_i),
        history=history,
    )


def train_from_h5(
    data_path: PathLike,
    target: str,
    bundle_dir: PathLike,
    hdf_key: Optional[str] = None,
    **kwargs: Any,
) -> TrainResult:
    """
    Load a table from disk and run train_from_dataframe.

    Parameters
    ----------
    hdf_key :
        Optional HDF5 table/group name. Must not be forwarded to
        ``train_from_dataframe``.
    """
    df = load_table(data_path, hdf_key=hdf_key)
    print(f"Loaded {data_path}: shape={df.shape}")
    return train_from_dataframe(df, target=target, bundle_dir=bundle_dir, **kwargs)
