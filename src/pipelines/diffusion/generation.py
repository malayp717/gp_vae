"""Sampling and reconstruction helpers for diffusion-family models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from torchvision.utils import make_grid, save_image

from src.config.loader import load_config
from src.data import get_cifar10_dataloaders
from src.latent import slerp
from src.models.base import encode_latent_mean
from src.runtime import get_device, get_model_type
from src.training.checkpoints import load_model_from_checkpoint

logger = logging.getLogger(__name__)


def _balanced_labels(num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_classes, device=device).repeat(num_samples // num_classes + 1)[:num_samples]


@torch.no_grad()
def generate_samples(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    num_samples: int = 64,
    output_dir: str | Path | None = None,
    steps: int | None = None,
) -> Path:
    device = get_device()
    model, ckpt_config = load_model_from_checkpoint(checkpoint_path, device)
    config = load_config(config_path) if config_path else ckpt_config
    model_type = config.get("model", {}).get("type", "sccd")
    base_out = Path(output_dir or config.get("paths", {}).get("output_dir", "./outputs"))
    out = base_out / get_model_type(config) / "samples"
    out.mkdir(parents=True, exist_ok=True)

    if model_type == "ddpm":
        steps = int(config.get("model", {}).get("ddpm_sample_steps", config.get("model", {}).get("diffusion_T", 1000)) if steps is None else steps)
        samples = model.sample(num_samples, device, steps=steps)
        nrow = max(1, int(num_samples ** 0.5))
    else:
        num_classes = int(config.get("model", {}).get("num_classes", 10))
        steps = int(config.get("model", {}).get("sample_steps", 1) if steps is None else steps)
        labels = _balanced_labels(num_samples, num_classes, device)
        samples = model.sample(num_samples, device, c=labels, steps=steps)
        nrow = min(num_classes, max(1, int(num_samples ** 0.5)))
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    save_path = out / "generated_samples.png"
    save_image(grid, save_path)
    logger.info("Saved %d diffusion samples -> %s", num_samples, save_path)
    return save_path


@torch.no_grad()
def generate_reconstructions(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    num_images: int = 16,
    output_dir: str | Path | None = None,
) -> Path:
    device = get_device()
    model, ckpt_config = load_model_from_checkpoint(checkpoint_path, device)
    config = load_config(config_path) if config_path else ckpt_config
    model_type = config.get("model", {}).get("type", "sccd")
    base_out = Path(output_dir or config.get("paths", {}).get("output_dir", "./outputs"))
    out = base_out / get_model_type(config) / "reconstructions"
    out.mkdir(parents=True, exist_ok=True)

    if model_type == "ddpm":
        steps = int(config.get("model", {}).get("ddpm_sample_steps", config.get("model", {}).get("diffusion_T", 1000)))
        samples = model.sample(num_images, device, steps=steps).cpu().clamp(0, 1)
        grid = make_grid(samples, nrow=max(1, int(num_images ** 0.5)), padding=2, normalize=False)
        save_path = out / "ddpm_samples_proxy.png"
        save_image(grid, save_path)
        logger.info("Saved %d DDPM samples to reconstruction slot -> %s", num_images, save_path)
        return save_path

    _, _, test_loader = get_cifar10_dataloaders(config)
    images, _ = next(iter(test_loader))
    images = images[:num_images].to(device)
    recon, _, _, _ = model(images)
    comparison = torch.cat([images, recon], dim=0)
    grid = make_grid(comparison, nrow=num_images, padding=2, normalize=False)
    save_path = out / "reconstructions.png"
    save_image(grid, save_path)
    logger.info("Saved %d diffusion reconstruction pairs -> %s", num_images, save_path)
    return save_path


@torch.no_grad()
def generate_interpolations(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    n_pairs: int = 4,
    n_steps: int = 10,
    output_dir: str | Path | None = None,
) -> Path:
    device = get_device()
    model, ckpt_config = load_model_from_checkpoint(checkpoint_path, device)
    config = load_config(config_path) if config_path else ckpt_config
    model_type = config.get("model", {}).get("type", "sccd")
    base_out = Path(output_dir or config.get("paths", {}).get("output_dir", "./outputs"))
    out = base_out / get_model_type(config) / "samples"
    out.mkdir(parents=True, exist_ok=True)

    if model_type == "ddpm":
        steps = int(config.get("model", {}).get("ddpm_sample_steps", config.get("model", {}).get("diffusion_T", 1000)))
        all_images = model.sample_interpolations(n_pairs=n_pairs, n_steps=n_steps, device=device, sample_steps=steps)
        grid = make_grid(all_images, nrow=n_steps, padding=2, normalize=False)
        save_path = out / "interpolations.png"
        save_image(grid, save_path)
        logger.info("Saved %d DDPM interpolation rows (%d steps each) -> %s", n_pairs, n_steps, save_path)
        return save_path

    _, _, test_loader = get_cifar10_dataloaders(config)
    images, _ = next(iter(test_loader))
    images = images[: 2 * n_pairs].to(device)
    mu = encode_latent_mean(model, images)
    rows: list[torch.Tensor] = []

    for index in range(n_pairs):
        z_start = mu[2 * index]
        z_end = mu[2 * index + 1]
        alphas = torch.linspace(0, 1, n_steps, device=device)
        z_interp = torch.stack([slerp(z_start, z_end, alpha.item()) for alpha in alphas])
        decoded = model.decode(z_interp)
        rows.append(decoded)

    all_images = torch.cat(rows, dim=0)
    grid = make_grid(all_images, nrow=n_steps, padding=2, normalize=False)
    save_path = out / "interpolations.png"
    save_image(grid, save_path)
    logger.info("Saved %d diffusion interpolation rows (%d steps each) -> %s", n_pairs, n_steps, save_path)
    return save_path


@torch.no_grad()
def save_validation_images(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    epoch: int,
    model_type: str,
    output_dir: Path,
    use_amp: bool,
    num_classes: int,
    n_images: int = 10,
    steps: int | None = None,
) -> None:
    """Save reconstruction pairs and class-balanced diffusion samples during validation."""
    model.eval()
    recon_dir = output_dir / "reconstructions"
    sample_dir = output_dir / "samples"
    recon_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "ddpm":
        steps = getattr(model, "sample_steps", 1000) if steps is None else steps
        samples = model.sample(n_images, device, steps=steps).cpu().clamp(0, 1)
        grid = make_grid(samples, nrow=max(1, int(n_images ** 0.5)), padding=2, normalize=False)
        save_image(grid, recon_dir / f"{model_type}_epoch_{epoch + 1:04d}.png")
        grid_s = grid
    else:
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

        steps = getattr(model, "sample_steps", 1) if steps is None else steps
        labels = _balanced_labels(n_images, num_classes, device)
        samples = model.sample(n_images, device, c=labels, steps=steps).cpu().clamp(0, 1)
        grid_s = make_grid(samples, nrow=min(num_classes, n_images), padding=2, normalize=False)
    save_image(grid_s, sample_dir / f"{model_type}_epoch_{epoch + 1:04d}.png")

    logger.info(
        "Saved %d diffusion validation images -> %s  |  %d samples -> %s",
        n_images,
        recon_dir.name,
        n_images,
        sample_dir.name,
    )
