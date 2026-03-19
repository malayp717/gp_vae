"""Sample, reconstruction, and interpolation generation for VAE-family models."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from vae.config.loader import load_config
from vae.data import get_cifar10_dataloaders
from vae.latent import slerp
from vae.models.base import encode_latent_mean
from vae.runtime import get_device, get_model_type
from vae.training.checkpoints import load_model_from_checkpoint

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_samples(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    num_samples: int = 64,
    output_dir: str | Path | None = None,
) -> Path:
    device = get_device()
    model, ckpt_config = load_model_from_checkpoint(checkpoint_path, device)
    config = load_config(config_path) if config_path else ckpt_config
    base_out = Path(output_dir or config.get("paths", {}).get("output_dir", "./outputs"))
    out = base_out / get_model_type(config) / "samples"
    out.mkdir(parents=True, exist_ok=True)

    samples = model.sample(num_samples, device)
    nrow = int(num_samples ** 0.5)
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    save_path = out / "generated_samples.png"
    save_image(grid, save_path)
    logger.info("Saved %d generated samples -> %s", num_samples, save_path)
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
    base_out = Path(output_dir or config.get("paths", {}).get("output_dir", "./outputs"))
    out = base_out / get_model_type(config) / "reconstructions"
    out.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_cifar10_dataloaders(config)
    images, _ = next(iter(test_loader))
    images = images[:num_images].to(device)
    recon, _, _, _ = model(images)
    comparison = torch.cat([images, recon], dim=0)
    grid = make_grid(comparison, nrow=num_images, padding=2, normalize=False)
    save_path = out / "reconstructions.png"
    save_image(grid, save_path)
    logger.info("Saved %d reconstruction pairs -> %s", num_images, save_path)
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
    base_out = Path(output_dir or config.get("paths", {}).get("output_dir", "./outputs"))
    out = base_out / get_model_type(config) / "samples"
    out.mkdir(parents=True, exist_ok=True)

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
    logger.info("Saved %d interpolation rows (%d steps each) -> %s", n_pairs, n_steps, save_path)
    return save_path
