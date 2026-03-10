from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .backbones import CNNBackbone, TransformerBackbone
from .fusion import SingleLayerFusion


class FaultDiagnosisNet(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        self.input_mode = data_cfg["input_mode"]
        self.backbone_name = model_cfg["backbone"]["name"]

        # project to common token dim=192 for transformer path
        if self.input_mode == "two_channel":
            self.input_proj = nn.Linear(63 * 2, 192)
        elif self.input_mode == "split_dual":
            self.input_proj = nn.Linear(63 * 2, 192)
        elif self.input_mode == "concat":
            self.input_proj = nn.Linear(126, 192)
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        fusion_cfg = model_cfg["fusion"]
        self.fusion = None
        if fusion_cfg["enabled"]:
            self.fusion = SingleLayerFusion(
                d_model=fusion_cfg["d_model"],
                nhead=fusion_cfg["nhead"],
                dim_feedforward=fusion_cfg["dim_feedforward"],
                dropout=fusion_cfg["dropout"],
                batch_first=fusion_cfg["batch_first"],
            )

        bb_cfg = model_cfg["backbone"]
        if self.backbone_name == "transformer":
            t_cfg = bb_cfg["transformer"]
            self.backbone = TransformerBackbone(
                d_model=t_cfg["d_model"],
                nhead=t_cfg["nhead"],
                dim_feedforward=t_cfg["dim_feedforward"],
                num_layers=t_cfg["num_layers"],
                dropout=bb_cfg["dropout"],
            )
            feat_dim = t_cfg["d_model"]
        elif self.backbone_name == "cnn":
            self.cnn_adapt = nn.Conv2d(2, 2, kernel_size=1)
            self.backbone = CNNBackbone(in_channels=2, base_channels=bb_cfg["cnn"]["base_channels"])
            feat_dim = self.backbone.out_dim
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        hidden1, hidden2 = model_cfg["classifier"]["hidden_dims"]
        drop = model_cfg["classifier"]["dropout"]
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden2, data_cfg["num_classes"]),
        )

        self.phys_head_in = nn.Sequential(nn.Linear(feat_dim, hidden2), nn.ReLU(inplace=True), nn.Linear(hidden2, 1))
        self.phys_head_out = nn.Sequential(nn.Linear(feat_dim, hidden2), nn.ReLU(inplace=True), nn.Linear(hidden2, 1))

    def _prepare_tokens(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.input_mode == "two_channel":
            x = batch["x"]  # [B,2,192,63]
            b, _, t, f = x.shape
            x = x.permute(0, 2, 1, 3).reshape(b, t, 2 * f)  # [B,192,126]
            return self.input_proj(x)  # [B,192,192]

        if self.input_mode == "split_dual":
            x1 = batch["x1"]
            x2 = batch["x2"]
            x = torch.cat([x1, x2], dim=-1)  # [B,192,126]
            return self.input_proj(x)

        x = batch["x"]  # [B,192,126]
        return self.input_proj(x)

    def _prepare_cnn(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.input_mode == "two_channel":
            return batch["x"]
        if self.input_mode == "split_dual":
            x = torch.stack([batch["x1"], batch["x2"]], dim=1)  # [B,2,192,63]
            return x
        x = batch["x"]
        x = x.reshape(x.shape[0], 2, 192, 63)
        return x

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.backbone_name == "transformer":
            tokens = self._prepare_tokens(batch)
            if self.fusion is not None:
                tokens = self.fusion(tokens)
            feat = self.backbone(tokens)
        else:
            x = self._prepare_cnn(batch)
            x = self.cnn_adapt(x)
            feat = self.backbone(x)

        logits = self.classifier(feat)
        phi_in_pred = self.phys_head_in(feat).squeeze(-1)
        phi_out_pred = self.phys_head_out(feat).squeeze(-1)

        return {
            "logits": logits,
            "phi_in_pred": phi_in_pred,
            "phi_out_pred": phi_out_pred,
        }
