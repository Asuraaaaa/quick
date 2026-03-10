from __future__ import annotations

from typing import Dict

import torch

from .net import FaultDiagnosisNet


def _load_pretrained(model: torch.nn.Module, transfer_cfg: Dict) -> None:
    if not transfer_cfg.get("enabled", False):
        return
    path = transfer_cfg.get("pretrained_path", "")
    if not path:
        raise ValueError("transfer.enabled=True but pretrained_path is empty")

    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(transfer_cfg.get("strict", False)))
    if missing:
        print(f"[Transfer] Missing keys: {missing}")
    if unexpected:
        print(f"[Transfer] Unexpected keys: {unexpected}")


def _freeze_backbone_if_needed(model: FaultDiagnosisNet, transfer_cfg: Dict) -> None:
    if not transfer_cfg.get("freeze_backbone", False):
        return
    for p in model.backbone.parameters():
        p.requires_grad = False
    if model.fusion is not None:
        for p in model.fusion.parameters():
            p.requires_grad = False
    print("[Transfer] Backbone and fusion are frozen.")


def build_model(cfg: Dict) -> FaultDiagnosisNet:
    model = FaultDiagnosisNet(cfg)
    transfer_cfg = cfg.get("transfer", {})
    _load_pretrained(model, transfer_cfg)
    _freeze_backbone_if_needed(model, transfer_cfg)
    return model
