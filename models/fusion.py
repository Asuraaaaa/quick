from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .encoder_layer import EncoderLayer, EncoderLayerDCA, EncoderLayerCA

class SingleLayerFusion(nn.Module):
    """One-layer TransformerEncoder for feature fusion."""

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float):
        super().__init__()
        self.layer = EncoderLayer(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            drop_prob=drop_prob
        )
        # self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SingleLayerFusion_DCA(nn.Module):
    """One-layer TransformerEncoder for feature fusion."""

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float):
        super().__init__()
        self.layer = EncoderLayerDCA(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            drop_prob=drop_prob
        )
        # self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.layer(x0, x1)


class SingleLayerFusion_CA(nn.Module):
    """One-layer TransformerEncoder for feature fusion."""

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float):
        super().__init__()
        self.layer = EncoderLayerCA(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            drop_prob=drop_prob
        )
        # self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.layer(x0, x1)

if __name__ == "__main__":
    # 简单测试
    attn = SingleLayerFusion(d_model=256, ffn_hidden=256, n_head=4, drop_prob=0.1)
    total_params = sum(p.numel() for p in attn.parameters())
    print(f"Total parameters of cross-attn: {total_params}")
    print(attn)
    dca_attn = SingleLayerFusion_DCA(d_model=256, ffn_hidden=256, n_head=4, drop_prob=0.1)
    total_params = sum(p.numel() for p in dca_attn.parameters())
    print(f"Total parameters of dca_attn: {total_params}")
    print(dca_attn)
    x0 = torch.randn(2, 129, 128)
    x1 = torch.randn(2, 129, 128)
    x = torch.randn(2, 129, 256)
    cross_out = dca_attn(x0, x1)
    attn_out = attn(x)

    print(f"Output shape: {attn_out.shape}")  # 应该是 (2, 63, 256)

    print(f"Output shape: {cross_out.shape}")  # 应该是 (2, 63, 256)
