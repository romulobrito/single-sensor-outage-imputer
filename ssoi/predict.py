"""
Inference API for single-sensor outage virtual sensing.

Deployment pattern: auxiliaries available, target missing
  t_hat = f_theta(x_scaled, t_tilde=0, m_t=1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from ssoi.bundle import ImputerBundle, load_bundle

ArrayLike = Union[np.ndarray, pd.DataFrame, Sequence[Sequence[float]]]
PathLike = Union[str, Path]


class VirtualSensor:
    """
    Load a TCDR bundle and impute the target under outage conditions.
    """

    def __init__(self, bundle: ImputerBundle, device: str = "cpu") -> None:
        self.bundle = bundle
        self.device = device
        self.bundle.model.to(device)
        self.bundle.model.eval()

    @classmethod
    def from_bundle(cls, bundle_dir: PathLike, device: str = "cpu") -> "VirtualSensor":
        """Construct from a bundle directory path."""
        return cls(load_bundle(bundle_dir, device=device), device=device)

    @property
    def features(self) -> List[str]:
        """Ordered auxiliary feature names required at inference."""
        return list(self.bundle.features)

    @property
    def target_name(self) -> str:
        """Name of the reconstructed target tag."""
        return self.bundle.target_name

    def _frame_from_input(self, x: ArrayLike) -> pd.DataFrame:
        """Normalize caller input to a DataFrame with expected columns."""
        if isinstance(x, pd.DataFrame):
            missing = [c for c in self.features if c not in x.columns]
            if missing:
                raise KeyError(f"Input DataFrame missing columns: {missing}")
            return x.loc[:, self.features].copy()

        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got shape {arr.shape}")
        if arr.shape[1] != len(self.features):
            raise ValueError(
                f"Expected {len(self.features)} features, got {arr.shape[1]}"
            )
        return pd.DataFrame(arr, columns=self.features)

    def _impute_auxiliaries(self, frame: pd.DataFrame) -> np.ndarray:
        """Fill NaNs in auxiliaries using train-only means, then zeros."""
        means = self.bundle.train_feature_means
        filled = frame.copy()
        for col in self.features:
            filled[col] = filled[col].fillna(means[col])
        filled = filled.fillna(0.0)
        return filled.to_numpy(dtype=np.float64)

    def predict(
        self,
        x: ArrayLike,
        *,
        return_scaled: bool = False,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Predict target values in outage mode (m_t=1, t_tilde=0).

        Args:
            x: DataFrame with feature columns, or ndarray (n, n_features)
            return_scaled: if True, return predictions in scaled target space
            batch_size: max rows per forward pass (must be >= 1)

        Returns:
            1D numpy array of predictions (engineering units by default)
        """
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        frame = self._frame_from_input(x)
        x_raw = self._impute_auxiliaries(frame)
        x_scaled = self.bundle.scaler_x.transform(x_raw).astype(np.float32)

        n = int(x_scaled.shape[0])
        chunks: List[np.ndarray] = []
        step = int(batch_size)
        self.bundle.model.eval()
        with torch.no_grad():
            for start in range(0, n, step):
                stop = min(start + step, n)
                x_t = torch.from_numpy(x_scaled[start:stop]).to(self.device)
                rows = int(x_t.shape[0])
                t_zero = torch.zeros((rows, 1), dtype=torch.float32, device=self.device)
                m_one = torch.ones((rows, 1), dtype=torch.float32, device=self.device)
                y_hat_scaled = self.bundle.model(x_t, t_zero, m_one).cpu().numpy()
                chunks.append(y_hat_scaled.reshape(-1))

        if chunks:
            y_hat_scaled = np.concatenate(chunks, axis=0)
        else:
            y_hat_scaled = np.empty((0,), dtype=np.float32)

        if return_scaled:
            return y_hat_scaled.reshape(-1)

        y_hat = self.bundle.scaler_y.inverse_transform(y_hat_scaled.reshape(-1, 1))
        return y_hat.reshape(-1)

    def predict_dict_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> List[float]:
        """
        Convenience API for ecosystem messages: list of feature dicts -> list of floats.
        """
        frame = pd.DataFrame(list(rows))
        preds = self.predict(frame)
        return [float(v) for v in preds]

    def info(self) -> Dict[str, Any]:
        """Return a small status dict useful for health checks."""
        return {
            "target_name": self.target_name,
            "n_features": len(self.features),
            "features": self.features,
            "device": self.device,
            "manifest": dict(self.bundle.manifest),
            "parameter_count": self.bundle.model.parameter_count(),
        }
