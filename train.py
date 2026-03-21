from __future__ import annotations
import os
import re
import copy
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from data.dataset import build_dataloaders
from losses.physics_loss import MultiTaskPhysicsLoss, CELoss, compute_norm_stats_from_csv, MultiTaskPhysicsDANNLoss
from losses.domain_generalization import coral_loss, mmd_loss
from utils.visualization import (
    plot_tsne,
    plot_tsne_fused,
    plot_confusion_matrix,
    plot_confusion_matrix_fused,
    plot_history_metrics_separate,
)
from models.build_model import build_model
from utils.config import load_config
from utils.metrics import accuracy, macro_f1, acc_elff
from utils.seed import set_seed

import math


def grl_lambda_schedule(epoch: int, max_epoch: int) -> float:
    p = epoch / max(max_epoch - 1, 1)
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0


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


def run_one_epoch(model, criterion, loader, optimizer, device, cfg, epoch, scaler=None, training=True):
    model.train(training)
    dg_cfg = cfg.get("dg", {})
    dg_method = str(dg_cfg.get("method", "none")).lower()
    lambda_dg = float(dg_cfg.get("lambda_dg", 0.0))
    mmd_kernel_mul = float(dg_cfg.get("mmd", {}).get("kernel_mul", 2.0))
    mmd_kernel_num = int(dg_cfg.get("mmd", {}).get("kernel_num", 5))

    total_loss = 0.0
    total_ce = 0.0
    total_in = 0.0
    total_out = 0.0
    total_center = 0.0
    total_domain = 0.0
    total_tn_compact = 0.0
    total_dg = 0.0

    all_logits = []
    all_targets = []

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        batch["_epoch"] = epoch

        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"

        with autocast(enabled=amp_enabled):
            outputs = model(batch)
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]
            l_dg = torch.tensor(0.0, device=device)

            # 可选域泛化正则：CORAL / MMD（MixStyle 在模型内部生效）
            if dg_method == "coral" and lambda_dg > 0.0:
                l_dg = coral_loss(outputs["feat"], batch["id_cylinder"])
                loss = loss + lambda_dg * l_dg
            elif dg_method == "mmd" and lambda_dg > 0.0:
                l_dg = mmd_loss(outputs["feat"], batch["id_cylinder"], kernel_mul=mmd_kernel_mul, kernel_num=mmd_kernel_num)
                loss = loss + lambda_dg * l_dg

            loss_dict["l_dg"] = l_dg.detach()

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

        total_loss += float(loss.item())
        total_ce += float(loss_dict["ce_loss"].item())
        total_in += float(loss_dict["l_in"].item())
        total_out += float(loss_dict["l_out"].item())
        total_domain += float(loss_dict.get("l_domain", torch.tensor(0.0, device=device)).item())
        total_center += float(loss_dict.get("l_center", torch.tensor(0.0, device=device)).item())
        total_tn_compact += float(loss_dict.get("l_tn_compact", torch.tensor(0.0, device=device)).item())
        total_dg += float(loss_dict.get("l_dg", torch.tensor(0.0, device=device)).item())

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
        "l_center": total_center / len(loader),
        "l_domain": total_domain / len(loader),
        "l_tn_compact": total_tn_compact / len(loader),
        "l_dg": total_dg / len(loader),
        "acc": accuracy(logits, targets),
        "merged_acc": acc_elff(logits, targets),
        "f1": macro_f1(logits, targets, num_classes=num_classes),
    }


@torch.no_grad()
def inference_on_loader(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    all_features = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        all_logits.append(outputs["logits"].detach().cpu())
        all_targets.append(batch["label"].detach().cpu())
        all_features.append(outputs["feat"].detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_features = torch.cat(all_features, dim=0)
    return all_logits, all_targets, all_features


def save_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_fold_dirs(fold_root: str | Path) -> List[Path]:
    fold_root = Path(fold_root)
    if not fold_root.exists():
        raise FileNotFoundError(f"fold_root does not exist: {fold_root}")

    fold_dirs = []
    for p in fold_root.iterdir():
        if p.is_dir():
            train_csv = p / "train.csv"
            val_csv = p / "val.csv"
            test_csv = p / "test.csv"
            if train_csv.exists() and val_csv.exists() and test_csv.exists():
                fold_dirs.append(p)

    if len(fold_dirs) == 0:
        raise RuntimeError(f"在 {fold_root} 下未找到包含 train.csv/val.csv/test.csv 的 fold 目录")

    def extract_fold_num(path: Path):
        m = re.search(r"fold_(\d+)", path.name)
        return int(m.group(1)) if m else 10**9

    fold_dirs = sorted(fold_dirs, key=extract_fold_num)
    return fold_dirs


def plot_cross_fold_metrics(summary_df: pd.DataFrame, save_path: Path):
    metrics = ["acc", "merged_acc", "f1", "loss"]
    x = np.arange(len(summary_df))
    width = 0.6

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.bar(x, summary_df[metric].values, width=width)
        plt.xticks(x, summary_df["fold_name"].values, rotation=30, ha="right")
        plt.title(metric)
        plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_fold_summary_table(fold_results: List[Dict], out_dir: Path):
    df = pd.DataFrame(fold_results)

    metric_cols = ["loss", "acc", "merged_acc", "f1"]
    stat_dict = {}
    for m in metric_cols:
        stat_dict[f"{m}_mean"] = float(df[m].mean())
        stat_dict[f"{m}_std"] = float(df[m].std(ddof=1)) if len(df) > 1 else 0.0

    summary = {
        "per_fold": fold_results,
        "aggregate": stat_dict
    }

    df.to_csv(out_dir / "kfold_test_summary.csv", index=False, encoding="utf-8-sig")
    save_json(summary, out_dir / "kfold_test_summary.json")
    plot_cross_fold_metrics(df, out_dir / "kfold_test_summary.png")

    print("\n========== K-FOLD TEST SUMMARY ==========")
    print(df[["fold_name", "loss", "acc", "merged_acc", "f1"]].to_string(index=False))
    print("-----------------------------------------")
    for m in metric_cols:
        mean_v = stat_dict[f"{m}_mean"]
        std_v = stat_dict[f"{m}_std"]
        print(f"{m}: {mean_v:.4f} ± {std_v:.4f}")
    print("=========================================\n")


def train_single_fold(cfg: Dict, fold_dir: Path, global_out_dir: Path, fold_index: int):
    fold_name = fold_dir.name
    out_dir = global_out_dir / fold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = fold_dir / "train.csv"
    val_csv = fold_dir / "val.csv"
    test_csv = fold_dir / "test.csv"

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
    )

    model = build_model(cfg).to(device)

    norm_stats = compute_norm_stats_from_csv(str(train_csv))
    # criterion = MultiTaskPhysicsLoss(cfg, norm_stats).to(device)
    criterion = MultiTaskPhysicsDANNLoss(cfg, norm_stats).to(device)
    # criterion = CELoss(cfg, norm_stats).to(device)

    optimizer = build_optimizer(cfg, model, criterion)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    best_metric = -1.0
    history = []

    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = run_one_epoch(
            model, criterion, train_loader, optimizer, device, cfg, epoch,
            scaler=scaler, training=True
        )
        val_metrics = run_one_epoch(
            model, criterion, val_loader, optimizer, device, cfg, epoch,
            scaler=None, training=False
        )
        scheduler.step()

        log_item = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(log_item)

        print(
            f"[{fold_name}] Epoch[{epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f}"
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
                "fold_name": fold_name,
            }
            torch.save(ckpt, out_dir / "best.pt")

    save_json(history, out_dir / "history_trainval.json")
    plot_history_metrics_separate(
        history_json_path=out_dir / "history_trainval.json",
        save_dir=out_dir,
    )

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    test_metrics = run_one_epoch(
        model, criterion, test_loader, optimizer, device, cfg, epoch=epochs,
        scaler=None, training=False
    )

    final_history = copy.deepcopy(history)
    final_history.append({"test": test_metrics})
    save_json(final_history, out_dir / "history.json")

    print(
        f"[{fold_name}] Test: "
        f"loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['acc']:.4f} "
        f"merged_acc={test_metrics['merged_acc']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )

    all_logits, all_targets, all_features = inference_on_loader(model, test_loader, device)

    np.save(out_dir / "logits.npy", all_logits.numpy())
    np.save(out_dir / "features.npy", all_features.numpy())
    np.save(out_dir / "targets.npy", all_targets.numpy())

    plot_tsne(
        X=all_features,
        y=all_targets,
        unique_classes=["N", "L1", "L2", "L3", "L2L3"],
        save_path=out_dir / "tsne_plot.jpg",
        font_path="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        figsize=(6.5, 6.0),
        dpi=400,
        random_state=42,
        perplexity=20,
        n_iter=1500,
        point_size=35,
        alpha=1.0,
    )

    plot_tsne_fused(
        features=all_features,
        labels=all_targets,
        save_path=out_dir / "tsne_plot_merged.jpg",
        figsize=(6.5, 6.0),
        dpi=400,
        random_state=42,
        perplexity=20,
        n_iter=1500,
        point_size=35,
        alpha=1.0,
    )

    cm = confusion_matrix(all_targets.numpy(), all_logits.argmax(dim=1).numpy())
    plot_confusion_matrix(
        cm=cm,
        class_names=["N", "L1", "L2", "L3", "L2L3"],
        save_path=out_dir / "confusion_matrix.jpg",
        font_path="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        figsize=(7.2, 6.4),
        dpi=400,
    )
    plot_confusion_matrix_fused(
        logits=all_logits,
        targets=all_targets,
        save_path=out_dir / "confusion_matrix_merged.jpg",
        dpi=400,
    )

    fold_result = {
        "fold_index": fold_index,
        "fold_name": fold_name,
        "best_val_f1": float(best_metric),
        "loss": float(test_metrics["loss"]),
        "acc": float(test_metrics["acc"]),
        "merged_acc": float(test_metrics["merged_acc"]),
        "f1": float(test_metrics["f1"]),
    }
    save_json(fold_result, out_dir / "test_metrics.json")
    return fold_result


def run_kfold_experiment(config_path: str, fold_root: str):
    cfg = load_config(config_path).raw
    set_seed(int(cfg["seed"]))

    fold_dirs = find_fold_dirs(fold_root)

    global_out_dir = Path(cfg["output_dir"])
    global_out_dir.mkdir(parents=True, exist_ok=True)

    config_backup = copy.deepcopy(cfg)
    save_json(config_backup, global_out_dir / "config_backup.json")

    fold_results = []
    for fold_index, fold_dir in enumerate(fold_dirs, start=1):
        print(f"\n{'=' * 80}")
        print(f"Start Fold {fold_index}: {fold_dir.name}")
        print(f"{'=' * 80}")

        fold_result = train_single_fold(
            cfg=cfg,
            fold_dir=fold_dir,
            global_out_dir=global_out_dir,
            fold_index=fold_index,
        )
        fold_results.append(fold_result)

    save_fold_summary_table(fold_results, global_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transformer.yaml")
    parser.add_argument(
        "--fold_root",
        type=str,
        default="/train1/gxc/HSFG_5/cylinder_splits",
        help="包含多个 fold_xxx 子目录的根目录，每个子目录下需有 train.csv / val.csv / test.csv"
    )
    args = parser.parse_args()

    run_kfold_experiment(args.config, args.fold_root)
