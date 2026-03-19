"""Latent-space interpolation helpers."""

from __future__ import annotations

import torch


def slerp(z1: torch.Tensor, z2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors."""
    z1_norm = z1 / (z1.norm() + 1e-8)
    z2_norm = z2 / (z2.norm() + 1e-8)
    omega = torch.acos(torch.clamp(torch.dot(z1_norm, z2_norm), -1.0, 1.0))
    if omega.abs() < 1e-6:
        return (1 - alpha) * z1 + alpha * z2
    sin_omega = torch.sin(omega)
    return (
        (torch.sin((1 - alpha) * omega) / sin_omega) * z1
        + (torch.sin(alpha * omega) / sin_omega) * z2
    )

