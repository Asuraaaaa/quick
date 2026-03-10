from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class PhysicsNormStats:
    phi_in_min: float
    phi_in_max: float
    phi_out_min: float
    phi_out_max: float


class MultiTaskPhysicsLoss(nn.Module):
    def __init__(self, cfg: Dict, norm_stats: PhysicsNormStats) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_in = float(cfg["loss"]["lambda_in"])
        self.lambda_out = float(cfg["loss"]["lambda_out"])
        self.alpha_in = float(cfg["loss"]["alpha_in"])
        self.alpha_out = float(cfg["loss"]["alpha_out"])
        self.eps = 1e-6
        self.norm_stats = norm_stats

        # 可学习参数 mu
        self.mu_in = nn.Parameter(torch.tensor(1.0))

    def _normalize(self, x: torch.Tensor, min_v: float, max_v: float) -> torch.Tensor:
        return (x - min_v) / (max_v - min_v + self.eps)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        phi_in_pred = outputs["phi_in_pred"]
        phi_out_pred = outputs["phi_out_pred"]

        labels = batch["label"]
        phi_in = batch["phi_in"]
        phi_out = batch["phi_out"]

        ce_loss = self.ce(logits, labels)

        phi_in_tilde = self._normalize(
            phi_in,
            self.norm_stats.phi_in_min,
            self.norm_stats.phi_in_max,
        )
        phi_out_tilde = self._normalize(
            phi_out,
            self.norm_stats.phi_out_min,
            self.norm_stats.phi_out_max,
        )

        # 假设: label 1/2 属于内泄类, label 3/4 属于外泄类, label 0 为正常类
        m_in = ((labels == 1) | (labels == 2)).float()
        m_out = ((labels == 3) | (labels == 4)).float()

        l_in = torch.mean(m_in * (phi_in_pred - self.mu_in * phi_in_tilde).pow(2) + self.alpha_in * (1 - m_in) * phi_in_pred.pow(2))
        l_out = torch.mean(m_out * (phi_out_pred - phi_out_tilde).pow(2) + self.alpha_out * (1 - m_out) * phi_out_pred.pow(2))

        total = ce_loss + self.lambda_in * l_in + self.lambda_out * l_out
        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "l_in": l_in.detach(),
            "l_out": l_out.detach(),
        }


def compute_norm_stats_from_csv(train_csv: str) -> PhysicsNormStats:
    import pandas as pd

    df = pd.read_csv(train_csv)
    return PhysicsNormStats(
        phi_in_min=float(df["phi_in"].min()),
        phi_in_max=float(df["phi_in"].max()),
        phi_out_min=float(df["phi_out"].min()),
        phi_out_max=float(df["phi_out"].max()),
    )
