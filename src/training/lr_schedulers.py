"""Learning-rate schedulers."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float,
    base_lr: float,
) -> LambdaLR:
    """Create a ``LambdaLR`` that linearly warms up then cosine-decays."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max(epoch / max(warmup_epochs, 1), min_lr / base_lr)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(cosine, min_lr / base_lr)

    return LambdaLR(optimizer, lr_lambda)

