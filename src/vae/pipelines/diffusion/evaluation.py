"""Evaluation runners for diffusion-family models."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm.auto import tqdm

from vae.config.loader import load_config
from vae.data import get_cifar10_dataloaders
from vae.losses import VAELoss, diagonal_kl_per_dim
from vae.models import get_kl_override, get_kl_per_dim
from vae.runtime import get_device
from vae.training.checkpoints import load_model_from_checkpoint

logger = logging.getLogger(__name__)


def _balanced_labels(num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_classes, device=device).repeat(num_samples // num_classes + 1)[:num_samples]


@torch.no_grad()
def compute_image_metrics(
    model: torch.nn.Module,
    model_type: str,
    loader: Any,
    device: torch.device,
    num_classes: int | None = None,
    num_fid_samples: int = 1024,
    fid_feature_dim: int = 2048,
    sample_steps: int | None = None,
) -> dict[str, float | None]:
    """Compute FID, SSIM, and PSNR for a diffusion-family model."""
    model.eval()
    fid_metric = FrechetInceptionDistance(feature=fid_feature_dim, normalize=True).to("cpu")
    sample_steps = getattr(model, "sample_steps", 1) if sample_steps is None else sample_steps
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to("cpu") if model_type == "sccd" else None
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to("cpu") if model_type == "sccd" else None

    collected_real = 0
    pbar_real = tqdm(loader, desc="Metrics (real+recon)", leave=False)
    for images, _ in pbar_real:
        if collected_real >= num_fid_samples:
            break
        images = images.to(device, non_blocking=True)
        if model_type == "sccd":
            recon, _, _, _ = model(images)
        images_cpu = images.cpu()
        if model_type == "sccd":
            assert ssim_metric is not None and psnr_metric is not None
            recon_cpu = recon.cpu().clamp(0, 1)
            ssim_metric.update(recon_cpu, images_cpu)
            psnr_metric.update(recon_cpu, images_cpu)
        fid_metric.update(images_cpu, real=True)
        collected_real += images.size(0)
        pbar_real.set_postfix(collected=collected_real)
    pbar_real.close()

    n_generated = 0
    batch_size = loader.batch_size or 64
    pbar_gen = tqdm(range(math.ceil(num_fid_samples / batch_size)), desc="Metrics (samples)", leave=False)
    for _ in pbar_gen:
        n = min(batch_size, num_fid_samples - n_generated)
        if model_type == "sccd":
            assert num_classes is not None
            labels = _balanced_labels(n, num_classes, device)
            samples = model.sample(n, device, c=labels, steps=sample_steps).cpu().clamp(0, 1)
        else:
            samples = model.sample(n, device, steps=sample_steps).cpu().clamp(0, 1)
        fid_metric.update(samples, real=False)
        n_generated += n
    pbar_gen.close()
    return {
        "fid": fid_metric.compute().item(),
        "ssim": ssim_metric.compute().item() if ssim_metric is not None else None,
        "psnr": psnr_metric.compute().item() if psnr_metric is not None else None,
    }


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    model_type: str,
    loader: Any,
    device: torch.device,
    beta: float = 1.0,
    use_amp: bool = False,
    criterion: VAELoss | None = None,
    consistency_weight: float = 1.0,
    consistency_k_steps: int = 5,
    return_kl_per_dim: bool = False,
) -> tuple[dict[str, float], torch.Tensor | None]:
    """Compute aggregate metrics over a loader and optionally KL-per-dim."""
    model.eval()
    use_reconstruction_loss = model_type == "sccd"
    if criterion is None and use_reconstruction_loss:
        criterion = VAELoss()
    if criterion is not None and criterion.has_discriminator and use_reconstruction_loss:
        criterion.discriminator.eval()  # type: ignore[union-attr]

    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "diffusion_loss": 0.0,
        "consistency_loss": 0.0,
        "noise_loss": 0.0,
        "lpips_loss": 0.0,
        "adv_g_loss": 0.0,
        "d_loss": 0.0,
    }
    n_batches = 0
    n_images = 0
    kl_per_dim_sum: torch.Tensor | None = None

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp):
            if model_type == "sccd":
                assert criterion is not None
                labels = labels.to(device, non_blocking=True)
                out = model.diffusion_forward(images, labels, k_steps=consistency_k_steps)
                vae_loss, comps = criterion(
                    out["recon"],
                    images,
                    out["mu"],
                    out["log_var"],
                    beta,
                    kl_override=get_kl_override(model),
                )
                consistency_loss = out["loss_cons"]
                diffusion_loss = consistency_loss
                noise_loss = torch.tensor(0.0, device=images.device)
                loss = vae_loss + consistency_weight * consistency_loss
            else:
                out = None
                loss = model.training_loss(images)
                comps = {
                    "recon": torch.tensor(0.0, device=images.device),
                    "kl": torch.tensor(0.0, device=images.device),
                    "lpips": torch.tensor(0.0, device=images.device),
                    "adv_g": torch.tensor(0.0, device=images.device),
                }
                consistency_loss = torch.tensor(0.0, device=images.device)
                diffusion_loss = loss
                noise_loss = loss

        running["loss"] += loss.item()
        running["recon_loss"] += comps["recon"].item()
        running["kl_loss"] += comps["kl"].item()
        running["diffusion_loss"] += diffusion_loss.item()
        running["consistency_loss"] += consistency_loss.item()
        running["noise_loss"] += noise_loss.item()
        running["lpips_loss"] += comps["lpips"].item()
        running["adv_g_loss"] += comps["adv_g"].item()
        if criterion is not None and criterion.has_discriminator and out is not None:
            running["d_loss"] += criterion.discriminator_loss(out["recon"], images).item()
        n_batches += 1
        n_images += images.size(0)

        if return_kl_per_dim and out is not None:
            batch_kl_per_dim = get_kl_per_dim(model)
            if batch_kl_per_dim is None:
                batch_kl_per_dim = diagonal_kl_per_dim(out["mu"], out["log_var"])
            if kl_per_dim_sum is None:
                kl_per_dim_sum = torch.zeros_like(batch_kl_per_dim, dtype=torch.float32, device="cpu")
            kl_per_dim_sum += batch_kl_per_dim.detach().to("cpu", dtype=torch.float32) * images.size(0)

    metrics = {key: value / max(n_batches, 1) for key, value in running.items()}
    if not return_kl_per_dim or kl_per_dim_sum is None or n_images == 0:
        return metrics, None
    return metrics, kl_per_dim_sum / float(n_images)


def run_validation(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, float]:
    """Load a checkpoint, run diffusion validation, and log aggregate metrics."""
    device = get_device()
    model, ckpt_config = load_model_from_checkpoint(checkpoint_path, device)
    config = load_config(config_path) if config_path else ckpt_config
    model_type = config.get("model", {}).get("type", "sccd")
    _, val_loader, test_loader = get_cifar10_dataloaders(config)
    loader = val_loader if split == "val" else test_loader
    use_amp = config.get("training", {}).get("mixed_precision", True) and device.type == "cuda"
    criterion = VAELoss.from_config(config, device) if model_type == "sccd" else None
    consistency_weight = float(config.get("loss", {}).get("consistency_weight", 1.0))
    consistency_k_steps = int(config.get("training", {}).get("consistency_k_steps", 5))
    num_classes = int(config.get("model", {}).get("num_classes", 10)) if model_type == "sccd" else None

    metrics, _ = evaluate_loader(
        model,
        model_type,
        loader,
        device,
        beta=1.0,
        use_amp=use_amp,
        criterion=criterion,
        consistency_weight=consistency_weight,
        consistency_k_steps=consistency_k_steps,
    )
    logger.info("=== %s metrics ===", split.upper())
    for key, value in metrics.items():
        logger.info("  %-18s %.4f", key, value)

    image_metrics = compute_image_metrics(
        model,
        model_type,
        loader,
        device,
        num_classes=num_classes,
        num_fid_samples=config.get("logging", {}).get("num_fid_samples", 1024),
    )
    logger.info("--- image metrics ---")
    for key, value in image_metrics.items():
        if value is None:
            logger.info("  %-18s -", key)
        else:
            logger.info("  %-18s %.4f", key, value)
    return metrics
