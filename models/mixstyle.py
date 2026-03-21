from __future__ import annotations

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """
    轻量 MixStyle：在特征维做统计量混合，提升跨域泛化。
    输入/输出均为 [B, D]。
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or x.dim() != 2:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        perm = torch.randperm(x.size(0), device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        lam = self.beta.sample((x.size(0), 1)).to(x.device, dtype=x.dtype)

        mu_mix = mu * lam + mu2 * (1.0 - lam)
        sig_mix = sig * lam + sig2 * (1.0 - lam)
        return x_norm * sig_mix + mu_mix

