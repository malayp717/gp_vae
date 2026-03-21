"""Training-time reconstruction and sample visualizations for VAE-family models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from torchvision.utils import make_grid, save_image

logger = logging.getLogger(__name__)


@torch.no_grad()
def save_validation_images(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    epoch: int,
    model_type: str,
    output_dir: Path,
    use_amp: bool,
    n_images: int = 10,
    n_samples: int | None = None,
) -> None:
    """Save reconstruction pairs and prior samples during validation."""
    model.eval()
    recon_dir = output_dir / "reconstructions"
    sample_dir = output_dir / "samples"
    recon_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    originals: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp):
            recon, _, _, _ = model(images)
        remaining = n_images - len(originals)
        originals.extend(images[:remaining].cpu())
        reconstructions.extend(recon[:remaining].cpu().clamp(0, 1))
        if len(originals) >= n_images:
            break

    orig_tensor = torch.stack(originals[:n_images])
    recon_tensor = torch.stack(reconstructions[:n_images])
    paired = torch.cat([orig_tensor, recon_tensor], dim=0)
    grid = make_grid(paired, nrow=n_images, padding=2, normalize=False)
    save_image(grid, recon_dir / f"{model_type}_epoch_{epoch + 1:04d}.png")

    sample_count = n_images if n_samples is None else int(n_samples)
    samples = model.sample(sample_count, device).cpu().clamp(0, 1)
    grid_cols = min(5, sample_count)
    grid_s = make_grid(samples, nrow=grid_cols, padding=2, normalize=False)
    save_image(grid_s, sample_dir / f"{model_type}_epoch_{epoch + 1:04d}.png")

    logger.info(
        "Saved %d reconstruction pairs -> %s  |  %d samples -> %s",
        n_images,
        recon_dir.name,
        sample_count,
        sample_dir.name,
    )
