# Copied from E01 code/e01/loss.py; only the docstring wording changed (PREREG §4).
"""Penalty-reduced pixel-wise focal loss (CenterNet, Zhou et al. 2019), PREREG §3: alpha 2, beta 4,
normalised by the number of peak pixels (target == 1). Logits in, computed in float32."""
from __future__ import annotations

import torch
from torch import nn

ALPHA = 2.0
BETA = 4.0
PRIOR = 0.01             # output bias init = log(p / (1 - p)) = -4.595


class HeatmapFocalLoss(nn.Module):
    def __init__(self, alpha: float = ALPHA, beta: float = BETA, eps: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        if target.shape != logits.shape:
            raise ValueError(f"target {tuple(target.shape)} != logits {tuple(logits.shape)}")
        p = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        # log-sigmoid forms are numerically stable for large |logits|
        log_p = nn.functional.logsigmoid(logits)
        log_1mp = nn.functional.logsigmoid(-logits)
        pos = target >= 1.0 - 1e-6
        pos_loss = -((1 - p) ** self.alpha) * log_p * pos
        neg_loss = -((1 - target) ** self.beta) * (p ** self.alpha) * log_1mp * (~pos)
        n_pos = pos.sum().clamp(min=1).float()
        return (pos_loss.sum() + neg_loss.sum()) / n_pos
