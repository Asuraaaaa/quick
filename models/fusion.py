from __future__ import annotations

import torch
from torch import nn


class SingleLayerFusion(nn.Module):
    """One-layer TransformerEncoder for feature fusion."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, batch_first: bool = True):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
