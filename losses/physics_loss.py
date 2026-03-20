from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhysicsNormStats:
    phi_in_min: float
    phi_in_max: float
    phi_out_min: float
    phi_out_max: float


class CELoss(nn.Module):
    def __init__(self, cfg: Dict, norm_stats: PhysicsNormStats) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        labels = batch["label"]

        ce_loss = self.ce(logits, labels)

        total = ce_loss
        device = logits.device
        zero = torch.tensor(0.0, device=device)
        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "l_in": zero,
            "l_out": zero,
            "l_center": zero,
            "l_center_in": zero,
            "l_center_out": zero,
        }


class SingleBranchCenterConsistencyLoss(nn.Module):
    """
    单个分支的油缸类中心一致性约束
    输入:
        feats:   [B, D]
        labels:  [B]
        domains: [B]
    """

    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        feat_dim: int,
        distance: str = "sq_l2",
        momentum: float = 0.9,
        normalize: bool = False,
        eps: float = 1e-12,
        detach_update: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        assert distance in {"l2", "sq_l2", "l1", "cosine", "smooth_l1"}
        assert reduction in {"mean", "sum"}

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.feat_dim = feat_dim
        self.distance = distance
        self.momentum = momentum
        self.normalize = normalize
        self.eps = eps
        self.detach_update = detach_update
        self.reduction = reduction

        # [C, M, D]
        self.register_buffer(
            "center_bank",
            torch.zeros(num_classes, num_domains, feat_dim)
        )
        # [C, M]
        self.register_buffer(
            "center_mask",
            torch.zeros(num_classes, num_domains, dtype=torch.bool)
        )

    def _prepare_features(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 2:
            raise ValueError(f"feats must be [B,D], got {tuple(feats.shape)}")

        B, D = feats.shape
        if D != self.feat_dim:
            raise ValueError(f"feat_dim mismatch: got {D}, expected {self.feat_dim}")

        if self.normalize:
            feats = F.normalize(feats, p=2, dim=-1)

        return feats

    def _pair_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a, b: [..., D]
        return: [...]
        """
        if self.distance == "l2":
            return torch.sqrt(((a - b) ** 2).sum(dim=-1) + self.eps)
        elif self.distance == "sq_l2":
            return ((a - b) ** 2).sum(dim=-1)
        elif self.distance == "l1":
            return (a - b).abs().sum(dim=-1)
        elif self.distance == "cosine":
            a_n = F.normalize(a, p=2, dim=-1)
            b_n = F.normalize(b, p=2, dim=-1)
            return 1.0 - (a_n * b_n).sum(dim=-1)
        elif self.distance == "smooth_l1":
            return F.smooth_l1_loss(a, b, reduction="none").sum(dim=-1)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def update_bank(self, feats: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor):
        if self.detach_update:
            feats = feats.detach()

        unique_classes = labels.unique()

        for c in unique_classes:
            c_idx = int(c.item())
            mask_c = (labels == c)
            domains_c = domains[mask_c].unique()

            for d in domains_c:
                d_idx = int(d.item())
                mask_cd = mask_c & (domains == d)

                if mask_cd.sum() == 0:
                    continue

                batch_center = feats[mask_cd].mean(dim=0)  # [D]

                if not self.center_mask[c_idx, d_idx]:
                    self.center_bank[c_idx, d_idx] = batch_center
                    self.center_mask[c_idx, d_idx] = True
                else:
                    self.center_bank[c_idx, d_idx] = (
                        self.momentum * self.center_bank[c_idx, d_idx]
                        + (1.0 - self.momentum) * batch_center
                    )

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        feats = self._prepare_features(feats)
        self.update_bank(feats, labels, domains)

        device = feats.device
        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0

        unique_classes = labels.unique()

        for c in unique_classes:
            c_idx = int(c.item())

            valid_domains = torch.where(self.center_mask[c_idx])[0]
            if len(valid_domains) < 2:
                continue

            domain_centers = self.center_bank[c_idx, valid_domains, :]  # [M_c, D]

            if self.normalize:
                domain_centers = F.normalize(domain_centers, p=2, dim=-1)

            global_center = domain_centers.mean(dim=0, keepdim=True)  # [1, D]
            if self.normalize:
                global_center = F.normalize(global_center, p=2, dim=-1)

            dists = self._pair_distance(domain_centers, global_center.expand_as(domain_centers))

            if self.reduction == "mean":
                class_loss = dists.mean()
            else:
                class_loss = dists.sum()

            total_loss = total_loss + class_loss
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss / valid_count


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

        # 第四章原有可学习参数
        self.mu_in = nn.Parameter(torch.tensor(1.0))

        # ===== 第五章配置 =====
        loss_cfg = cfg.get("loss", {})
        self.use_center_consistency = bool(loss_cfg.get("use_center_consistency", False))
        self.lambda_center = float(loss_cfg.get("lambda_center", 0.0))
        self.lambda_center_in = float(loss_cfg.get("lambda_center_in", 1.0))
        self.lambda_center_out = float(loss_cfg.get("lambda_center_out", 1.0))
        self.center_warmup_epochs = int(loss_cfg.get("center_warmup_epochs", 0))

        self.center_in_loss_fn: Optional[SingleBranchCenterConsistencyLoss] = None
        self.center_out_loss_fn: Optional[SingleBranchCenterConsistencyLoss] = None

        if self.use_center_consistency:
            num_classes = int(cfg["data"]["num_classes"])
            num_domains = int(loss_cfg["num_domains"])
            center_feat_dim = int(loss_cfg["center_feat_dim"])
            center_distance = str(loss_cfg.get("center_distance", "sq_l2"))
            center_momentum = float(loss_cfg.get("center_momentum", 0.9))
            center_normalize = bool(loss_cfg.get("center_normalize", False))
            center_reduction = str(loss_cfg.get("center_reduction", "mean"))

            self.center_in_loss_fn = SingleBranchCenterConsistencyLoss(
                num_classes=num_classes,
                num_domains=num_domains,
                feat_dim=center_feat_dim,
                distance=center_distance,
                momentum=center_momentum,
                normalize=center_normalize,
                reduction=center_reduction,
            )

            self.center_out_loss_fn = SingleBranchCenterConsistencyLoss(
                num_classes=num_classes,
                num_domains=num_domains,
                feat_dim=center_feat_dim,
                distance=center_distance,
                momentum=center_momentum,
                normalize=center_normalize,
                reduction=center_reduction,
            )

    def _normalize(self, x: torch.Tensor, min_v: float, max_v: float) -> torch.Tensor:
        return (x - min_v) / (max_v - min_v + self.eps)

    def _compute_center_lambda(self, batch: Dict[str, torch.Tensor]) -> float:
        if self.center_warmup_epochs <= 0:
            return self.lambda_center

        epoch = batch.get("_epoch", None)
        if epoch is None:
            return self.lambda_center

        cur_lambda = self.lambda_center * min(float(epoch) / float(self.center_warmup_epochs), 1.0)
        return cur_lambda

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        phi_in_pred = outputs["phi_in_pred"].squeeze(-1)   # [B,1] -> [B]
        phi_out_pred = outputs["phi_out_pred"].squeeze(-1) # [B,1] -> [B]

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

        # 保持你原有逻辑不变
        m_out = ((labels == 2) | (labels == 3) | (labels == 4)).float()
        m_in = (labels == 1).float()

        l_in = torch.mean(
            m_in * (phi_in_pred - self.mu_in * phi_in_tilde).pow(2)
            + self.alpha_in * (1 - m_in) * phi_in_pred.pow(2)
        )
        l_out = torch.mean(
            m_out * (phi_out_pred - phi_out_tilde).pow(2)
            + self.alpha_out * (1 - m_out) * phi_out_pred.pow(2)
        )

        total = ce_loss + self.lambda_in * l_in + self.lambda_out * l_out

        device = logits.device
        zero = torch.tensor(0.0, device=device)
        l_center_in = zero
        l_center_out = zero
        l_center = zero

        # ===== 第五章中心一致性 =====
        if self.use_center_consistency:
            domains = batch.get("id_cylinder", None)
            phi_in_feat = outputs.get("phi_in_feat", None)
            phi_out_feat = outputs.get("phi_out_feat", None)

            if (domains is not None) and (phi_in_feat is not None) and (phi_out_feat is not None):
                cur_lambda_center = self._compute_center_lambda(batch)

                if cur_lambda_center > 0.0:
                    l_center_in = self.center_in_loss_fn(
                        feats=phi_in_feat,
                        labels=labels,
                        domains=domains,
                    )
                    l_center_out = self.center_out_loss_fn(
                        feats=phi_out_feat,
                        labels=labels,
                        domains=domains,
                    )

                    l_center = (
                        self.lambda_center_in * l_center_in
                        + self.lambda_center_out * l_center_out
                    )

                    total = total + cur_lambda_center * l_center

        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "l_in": l_in.detach(),
            "l_out": l_out.detach(),
            "l_center": l_center.detach(),
            "l_center_in": l_center_in.detach(),
            "l_center_out": l_center_out.detach(),
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


class MultiTaskPhysicsDANNLoss(nn.Module):
    def __init__(self, cfg: Dict, norm_stats: PhysicsNormStats) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.domain_ce = nn.CrossEntropyLoss()

        self.lambda_in = float(cfg["loss"]["lambda_in"])
        self.lambda_out = float(cfg["loss"]["lambda_out"])
        self.lambda_domain = float(cfg["loss"].get("lambda_domain", 0.0))

        self.alpha_in = float(cfg["loss"]["alpha_in"])
        self.alpha_out = float(cfg["loss"]["alpha_out"])

        self.eps = 1e-6
        self.norm_stats = norm_stats

        self.mu_in = nn.Parameter(torch.tensor(1.0))

        # 目标域 normal 物理锚定（轻量 running prototype）
        loss_cfg = cfg.get("loss", {})
        self.enable_tn_compact = bool(loss_cfg.get("enable_tn_compact", False))
        self.lambda_tn_compact = float(loss_cfg.get("lambda_tn_compact", 0.0))
        self.tn_compact_momentum = float(loss_cfg.get("tn_compact_momentum", 0.9))
        self.tn_compact_eps = 1e-8
        self.register_buffer("tn_running_proto", torch.zeros(1))
        self.register_buffer("tn_has_proto", torch.tensor(False, dtype=torch.bool))

    def _normalize(self, x: torch.Tensor, min_v: float, max_v: float) -> torch.Tensor:
        return (x - min_v) / (max_v - min_v + self.eps)

    @torch.no_grad()
    def _update_tn_running_proto(self, batch_proto: torch.Tensor) -> None:
        if (not self.tn_has_proto.item()) or (self.tn_running_proto.numel() != batch_proto.numel()):
            self.tn_running_proto = batch_proto.detach().clone()
            self.tn_has_proto.fill_(True)
            return

        self.tn_running_proto = (
            self.tn_compact_momentum * self.tn_running_proto
            + (1.0 - self.tn_compact_momentum) * batch_proto.detach()
        )

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        phi_in_pred = outputs["phi_in_pred"].squeeze(-1)   # [B]
        phi_out_pred = outputs["phi_out_pred"].squeeze(-1) # [B]
        domain_logits = outputs["domains"]           # [B, num_domains]

        labels = batch["label"]
        domains = batch["id_cylinder"]                     # [B]
        phi_in = batch["phi_in"]
        phi_out = batch["phi_out"]

        ce_loss = self.ce(logits, labels)

        phi_in_tilde = self._normalize(
            phi_in, self.norm_stats.phi_in_min, self.norm_stats.phi_in_max
        )
        phi_out_tilde = self._normalize(
            phi_out, self.norm_stats.phi_out_min, self.norm_stats.phi_out_max
        )

        # 你当前已有的类别逻辑保持不变
        m_out = ((labels == 2) | (labels == 3) | (labels == 4)).float()
        m_in = (labels == 1).float()

        l_in = torch.mean(
            m_in * (phi_in_pred - self.mu_in * phi_in_tilde).pow(2)
            + self.alpha_in * (1 - m_in) * phi_in_pred.pow(2)
        )
        l_out = torch.mean(
            m_out * (phi_out_pred - phi_out_tilde).pow(2)
            + self.alpha_out * (1 - m_out) * phi_out_pred.pow(2)
        )

        l_domain = self.domain_ce(domain_logits, domains)

        device = logits.device
        l_tn_compact = torch.tensor(0.0, device=device)

        if self.enable_tn_compact and self.lambda_tn_compact > 0:
            feats = outputs.get("feat", None)
            tn_mask = batch.get("is_target_normal", None)
            if (feats is not None) and (tn_mask is not None):
                tn_mask = tn_mask.bool()
                if tn_mask.any():
                    tn_feats = feats[tn_mask]  # [N_tn, D]
                    batch_proto = tn_feats.mean(dim=0)  # [D]
                    self._update_tn_running_proto(batch_proto)

                    anchor_proto = self.tn_running_proto
                    if anchor_proto.ndim == 1:
                        anchor_proto = anchor_proto.unsqueeze(0)

                    l_tn_compact = ((tn_feats - anchor_proto) ** 2).sum(dim=-1).mean()

        total = (
            ce_loss
            + self.lambda_in * l_in
            + self.lambda_out * l_out
            + self.lambda_domain * l_domain
            + self.lambda_tn_compact * l_tn_compact
        )

        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "l_in": l_in.detach(),
            "l_out": l_out.detach(),
            "l_domain": l_domain.detach(),
            "l_tn_compact": l_tn_compact.detach(),
        }
