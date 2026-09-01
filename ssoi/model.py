"""
TCDR network: target-conditional denoising regressor.

Maps observability-aware input [x; t_tilde; m_t] to scalar target estimate.
Extracted from the paper research implementation (ConditionalDenoisingAE).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def tcdr_hidden_widths(n_features: int, latent_dim: int = 16) -> Tuple[int, int, int]:
    """
    Compute hidden widths from auxiliary feature count.

    Capacity uses in_dim = n_features + 2 (x plus t_tilde and m_t), matching
    the manuscript sizing rule.
    """
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")
    capacity_in_dim = int(n_features) + 2
    h1 = max(64, 8 * capacity_in_dim)
    h2 = max(32, 4 * capacity_in_dim)
    return h1, h2, int(latent_dim)


class TargetConditionalDenoisingRegressor(nn.Module):
    """
    Supervised scalar regressor for single-sensor outage recovery.

    Input channels:
      - x: auxiliary sensors (n_features)
      - t_tilde: observed target or 0 when missing (scaled space)
      - m_t: 1 if target unavailable, 0 if observed

    Output:
      - t_hat: predicted target in the same scaled space as training
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        self.n_features = int(n_features)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        h1, h2, latent = tcdr_hidden_widths(self.n_features, self.latent_dim)
        self.h1 = h1
        self.h2 = h2
        in_dim = self.n_features + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout / 2.0),
            nn.Linear(h2, latent),
            nn.BatchNorm1d(latent),
            nn.LeakyReLU(0.1),
            nn.Linear(latent, h2),
            nn.LeakyReLU(0.1),
            nn.Linear(h2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_tilde: torch.Tensor,
        m_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: shape (batch, n_features)
            t_tilde: shape (batch, 1)
            m_t: shape (batch, 1)
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (batch, n_features), got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"x has {x.shape[1]} features, model expects {self.n_features}"
            )
        inp = torch.cat([x, t_tilde, m_t], dim=1)
        return self.net(inp)

    def parameter_count(self) -> int:
        """Return number of trainable parameters."""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def config_dict(self) -> Dict[str, object]:
        """Serializable architecture metadata for bundle manifests."""
        return {
            "model_type": "tcdr",
            "n_features": self.n_features,
            "in_dim": self.n_features + 2,
            "h1": self.h1,
            "h2": self.h2,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "parameter_count": self.parameter_count(),
        }
