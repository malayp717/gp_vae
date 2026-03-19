"""Beta annealing schedules."""

from __future__ import annotations

import math
from typing import Any


class BetaScheduler:
    """Compute the KL weight (beta) at a given training step."""

    def __init__(self, config: dict[str, Any]) -> None:
        beta_cfg = config.get("beta_annealing", {})
        self.enabled = beta_cfg.get("enabled", True)
        self.schedule = beta_cfg.get("schedule", "linear")
        self.beta_min = beta_cfg.get("beta_min", 0.0)
        self.beta_max = beta_cfg.get("beta_max", 4.0)
        self.warmup_epochs = beta_cfg.get("warmup_epochs", 10)
        self.cycle_epochs = beta_cfg.get("cycle_epochs", 30)
        self.n_cycles = beta_cfg.get("n_cycles", 4)
        self.ratio_increase = beta_cfg.get("ratio_increase", 0.5)
        self._total_epochs = config.get("training", {}).get("epochs", 200)

    def __call__(self, epoch: int, step: int = 0, steps_per_epoch: int = 1) -> float:
        if not self.enabled:
            return self.beta_max
        progress = epoch + step / max(steps_per_epoch, 1)
        if self.schedule == "linear":
            return self._linear(progress)
        if self.schedule == "cosine":
            return self._cosine(progress)
        if self.schedule == "cyclical":
            return self._cyclical(progress)
        raise ValueError(f"Unknown beta schedule: {self.schedule}")

    def _linear(self, progress: float) -> float:
        if self.warmup_epochs <= 0:
            return self.beta_max
        t = min(progress / self.warmup_epochs, 1.0)
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def _cosine(self, progress: float) -> float:
        if self.warmup_epochs <= 0:
            return self.beta_max
        t = min(progress / self.warmup_epochs, 1.0)
        return self.beta_min + (self.beta_max - self.beta_min) * 0.5 * (1 - math.cos(math.pi * t))

    def _cyclical(self, progress: float) -> float:
        cycle_len = self.cycle_epochs
        if cycle_len <= 0:
            return self.beta_max
        tau = (progress % cycle_len) / cycle_len
        ramp = self.ratio_increase
        t = tau / ramp if tau <= ramp else 1.0
        return self.beta_min + (self.beta_max - self.beta_min) * t

