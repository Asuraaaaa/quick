from __future__ import annotations

from itertools import combinations
from typing import List

import torch
import torch.nn.functional as F


def _split_by_domain(feats: torch.Tensor, domains: torch.Tensor) -> List[torch.Tensor]:
    """按域切分特征，仅保留批次中存在的域。"""
    domain_feats: List[torch.Tensor] = []
    for d in domains.unique(sorted=True):
        cur = feats[domains == d]
        if cur.size(0) > 0:
            domain_feats.append(cur)
    return domain_feats


def coral_loss(feats: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    """
    多域 CORAL：对批次内各域两两协方差差异求平均。
    feats: [B, D], domains: [B]
    """
    device = feats.device
    domain_feats = _split_by_domain(feats, domains)
    if len(domain_feats) < 2:
        return torch.tensor(0.0, device=device)

    losses = []
    feat_dim = feats.size(1)
    denom = 4.0 * float(feat_dim * feat_dim)

    for f1, f2 in combinations(domain_feats, 2):
        if f1.size(0) < 2 or f2.size(0) < 2:
            continue
        c1 = _covariance(f1)
        c2 = _covariance(f2)
        losses.append(((c1 - c2) ** 2).sum() / denom)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    n = x.size(0)
    return x.t().matmul(x) / max(n - 1, 1)


def mmd_loss(feats: torch.Tensor, domains: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    """
    多域 MMD(RBF)：对批次内各域两两 MMD 求平均。
    """
    device = feats.device
    domain_feats = _split_by_domain(feats, domains)
    if len(domain_feats) < 2:
        return torch.tensor(0.0, device=device)

    losses = []
    for f1, f2 in combinations(domain_feats, 2):
        if f1.size(0) < 2 or f2.size(0) < 2:
            continue
        losses.append(_mmd_rbf(f1, f2, kernel_mul=kernel_mul, kernel_num=kernel_num))

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    xx = _gaussian_kernel(x, x, kernel_mul=kernel_mul, kernel_num=kernel_num)
    yy = _gaussian_kernel(y, y, kernel_mul=kernel_mul, kernel_num=kernel_num)
    xy = _gaussian_kernel(x, y, kernel_mul=kernel_mul, kernel_num=kernel_num)
    return xx.mean() + yy.mean() - 2.0 * xy.mean()


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    dist2 = torch.cdist(x, y, p=2) ** 2
    base_sigma = dist2.detach().mean()
    if base_sigma.item() <= 0:
        base_sigma = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    base_sigma = base_sigma / (kernel_mul ** (kernel_num // 2))

    kernels = []
    for i in range(kernel_num):
        sigma = base_sigma * (kernel_mul ** i)
        kernels.append(torch.exp(-dist2 / (sigma + 1e-6)))
    return sum(kernels) / float(kernel_num)

