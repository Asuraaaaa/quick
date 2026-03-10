from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = torch.argmax(logits, dim=1)
    f1_scores = []
    eps = 1e-8
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1)

    return torch.stack(f1_scores).mean().item()
