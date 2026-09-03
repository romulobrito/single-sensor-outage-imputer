#!/usr/bin/env python3
"""
Synthetic smoke test: train a tiny TCDR, save a bundle, reload, predict.

Does not use industrial data. Safe to run after pip install -e .
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssoi import (  # noqa: E402
    TargetConditionalDenoisingRegressor,
    VirtualSensor,
    save_bundle,
)
from ssoi.train import _drop_last_for_batchnorm  # noqa: E402


def _make_synthetic(
    n_samples: int = 800,
    n_features: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create correlated auxiliaries and a scalar target."""
    rng = np.random.default_rng(seed)
    features = [f"aux_{i:02d}" for i in range(n_features)]
    x = rng.normal(size=(n_samples, n_features))
    # Target depends mainly on first two auxiliaries plus noise
    y = 3.0 * x[:, 0] + 1.5 * x[:, 1] - 0.5 * x[:, 2] + rng.normal(scale=0.3, size=n_samples)
    return x.astype(np.float64), y.astype(np.float64), features


def _train_short(
    model: TargetConditionalDenoisingRegressor,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    epochs: int = 40,
    p_mask: float = 0.5,
    device: str = "cpu",
) -> None:
    """Minimal mask-weighted training loop for the smoke test."""
    ds = TensorDataset(
        torch.from_numpy(x_scaled.astype(np.float32)),
        torch.from_numpy(y_scaled.reshape(-1, 1).astype(np.float32)),
    )
    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        drop_last=_drop_last_for_batchnorm(len(ds), 64),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction="none")
    model.to(device)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            m_t = (torch.rand_like(yb) < p_mask).float()
            t_tilde = (1.0 - m_t) * yb
            opt.zero_grad()
            y_hat = model(xb, t_tilde, m_t)
            per = loss_fn(y_hat, yb)
            loss = ((1.0 + m_t) * per).mean()
            loss.backward()
            opt.step()
    model.eval()


def main() -> int:
    """Run end-to-end synthetic bundle create + predict."""
    bundle_dir = ROOT / "examples" / "synthetic_bundle"
    device = "cpu"

    x, y, features = _make_synthetic()
    n_train = int(0.8 * len(x))
    x_train, y_train = x[:n_train], y[:n_train]
    x_test = x[n_train:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train_s = scaler_x.fit_transform(x_train)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    means = {name: float(x_train[:, i].mean()) for i, name in enumerate(features)}

    model = TargetConditionalDenoisingRegressor(
        n_features=len(features),
        latent_dim=8,
        dropout=0.1,
    )
    _train_short(model, x_train_s, y_train_s, device=device)

    save_bundle(
        bundle_dir,
        features=features,
        train_feature_means=means,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        model=model,
        target_name="Target_Synthetic",
        manifest={"example": "smoke_synthetic", "seed": 42},
    )
    print(f"[ok] bundle written to {bundle_dir}")

    sensor = VirtualSensor.from_bundle(bundle_dir, device=device)
    preds = sensor.predict(x_test)
    print(f"[ok] predicted {preds.shape[0]} rows")
    print(f"[ok] pred mean={float(preds.mean()):.4f} std={float(preds.std()):.4f}")
    print(f"[ok] sensor.info={sensor.info()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
