from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

@dataclass
class Config:
    raw: Dict[str, Any]

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.raw.get(item, default)



def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(raw=cfg)


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

if __name__ == "__main__":
    cfg = load_config('/train1/gxc/HSFG/configs/config.yaml').raw
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters of model: {total_params}")
    x1 = torch.randn(1, 129, 63).to(device)
    x2 = torch.randn(1, 129, 63).to(device)
    batch = {"x1": x1, "x2": x2}
    cross_out = model(batch)
    # "logits": logits,
    # "phi_in_pred": phi_in_pred,
    # "phi_out_pred": phi_out_pred,
    print(f"Output shape: {cross_out["logits"].shape}")
    from thop import profile
    import torch.nn as nn
    flops, params = profile(model, inputs=(batch, ))
    print(f"FLOPs: {flops / 1e9:.2f} G") # 以 GFLOPs 为单位输出
