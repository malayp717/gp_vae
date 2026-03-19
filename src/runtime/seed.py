"""Randomness and reproducibility helpers."""

from __future__ import annotations

import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all backends."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

