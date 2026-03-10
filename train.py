from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast

from data.dataset import build_dataloaders
from losses.physics_loss import MultiTaskPhysicsLoss, compute_norm_stats_from_csv
from models.build_model import build_model
from utils.config import load_config
from utils.metrics import accuracy, macro_f1
from utils.seed import set_seed


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return moved


def build_optimizer(cfg: Dict, model: torch.nn.Module, criterion: torch.nn.Module) -> torch.optim.Optimizer:
    params = list(model.parameters()) + list(criterion.parameters())
    opt_cfg = cfg["optimizer"]
    if opt_cfg["name"].lower() == "adamw":
        return torch.optim.AdamW(params, lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"]))
    raise ValueError(f"Unsupported optimizer: {opt_cfg['name']}")


def build_scheduler(cfg: Dict, optimizer: torch.optim.Optimizer):
    s_cfg = cfg["scheduler"]
    if s_cfg["name"].lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(s_cfg["t_max"]),
            eta_min=float(s_cfg["eta_min"]),
        )
    raise ValueError(f"Unsupported scheduler: {s_cfg['name']}")


def run_one_epoch(model, criterion, loader, optimizer, device, cfg, scaler=None, training=True):
    model.train(training)
    total_loss = 0.0
    total_ce = 0.0
    total_in = 0.0
    total_out = 0.0
    all_logits = []
    all_targets = []

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
        with autocast(enabled=amp_enabled):
            outputs = model(batch)
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]

        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                if cfg["train"].get("grad_clip", 0.0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg["train"].get("grad_clip", 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                optimizer.step()

        total_loss += loss.item()
        total_ce += float(loss_dict["ce_loss"].item())
        total_in += float(loss_dict["l_in"].item())
        total_out += float(loss_dict["l_out"].item())
        all_logits.append(outputs["logits"].detach())
        all_targets.append(batch["label"].detach())

        if training and (step + 1) % int(cfg["train"]["log_interval"]) == 0:
            print(f"step={step+1}/{len(loader)} loss={loss.item():.4f}")

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    num_classes = int(cfg["data"]["num_classes"])

    return {
        "loss": total_loss / len(loader),
        "ce_loss": total_ce / len(loader),
        "l_in": total_in / len(loader),
        "l_out": total_out / len(loader),
        "acc": accuracy(logits, targets),
        "f1": macro_f1(logits, targets, num_classes=num_classes),
    }


def main(config_path: str) -> None:
    cfg = load_config(config_path).raw
    set_seed(int(cfg["seed"]))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    norm_stats = compute_norm_stats_from_csv(cfg["data"]["train_csv"])
    criterion = MultiTaskPhysicsLoss(cfg, norm_stats).to(device)

    optimizer = build_optimizer(cfg, model, criterion)
    scheduler = build_scheduler(cfg, optimizer)

    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    best_metric = -1.0
    history = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_one_epoch(model, criterion, train_loader, optimizer, device, cfg, scaler=scaler, training=True)
        val_metrics = run_one_epoch(model, criterion, val_loader, optimizer, device, cfg, scaler=None, training=False)
        scheduler.step()

        log_item = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(log_item)

        print(
            f"Epoch[{epoch}] "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        target_metric = val_metrics["f1"]
        if target_metric > best_metric:
            best_metric = target_metric
            ckpt = {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "config": cfg,
            }
            torch.save(ckpt, out_dir / "best.pt")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
