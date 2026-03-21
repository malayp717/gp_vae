"""Evaluation runners for diffusion-family models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm.auto import tqdm

from src.config.loader import load_config
from src.data import get_cifar10_dataloaders
from src.runtime import get_device
from src.training.checkpoints import load_model_from_checkpoint

logger = logging.getLogger(__name__)


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
    """Compute sample and reconstruction metrics for a diffusion-family model."""
    model.eval()
    fid_metric = FrechetInceptionDistance(feature=fid_feature_dim, normalize=True).to("cpu")
    sample_steps = getattr(model, "sample_steps", 1) if sample_steps is None else sample_steps
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to("cpu") if model_type == "sccd" else None
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to("cpu") if model_type == "sccd" else None

    collected_real = 0
    pbar_real = tqdm(loader, desc="Metrics (real)", leave=False)
    for images, _ in pbar_real:
        if collected_real >= num_fid_samples:
            break
        images = images.to(device, non_blocking=True)
        if model_type == "sccd" and ssim_metric is not None and psnr_metric is not None:
            recon = model.reconstruct(images).cpu().clamp(0, 1)
            images_cpu = images.cpu()
            ssim_metric.update(recon, images_cpu)
            psnr_metric.update(recon, images_cpu)
        else:
            images_cpu = images.cpu()
        fid_metric.update(images_cpu, real=True)
        collected_real += images.size(0)
        pbar_real.set_postfix(collected=collected_real)
    pbar_real.close()

    n_generated = 0
    batch_size = loader.batch_size or 64
    n_iters = (num_fid_samples + batch_size - 1) // batch_size
    pbar_gen = tqdm(range(n_iters), desc="Metrics (samples)", leave=False)
    for _ in pbar_gen:
        n = min(batch_size, num_fid_samples - n_generated)
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
    criterion: torch.nn.Module | None = None,
    consistency_weight: float = 1.0,
    consistency_k_steps: int = 5,
    return_kl_per_dim: bool = False,
) -> tuple[dict[str, float], torch.Tensor | None]:
    """Compute aggregate metrics over a loader."""
    model.eval()
    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "boundary_loss": 0.0,
        "rank_penalty": 0.0,
        "sigma_penalty": 0.0,
        "sigma_mean": 0.0,
        "sigma_min": 0.0,
        "diffusion_loss": 0.0,
        "noise_loss": 0.0,
    }
    n_batches = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp):
            if model_type == "sccd":
                out = model.training_loss(images, labels=labels)
                loss = out["total"]
                recon_loss = out["recon"]
                boundary_loss = out["boundary"]
                rank_penalty = out["rank_penalty"]
                sigma_penalty = out.get("sigma_penalty", torch.tensor(0.0, device=images.device))
                sigma_mean = out.get("sigma_mean", torch.tensor(0.0, device=images.device))
                sigma_min = out.get("sigma_min", torch.tensor(0.0, device=images.device))
                diffusion_loss = loss
                noise_loss = torch.tensor(0.0, device=images.device)
            else:
                loss = model.training_loss(images)
                recon_loss = torch.tensor(0.0, device=images.device)
                boundary_loss = torch.tensor(0.0, device=images.device)
                rank_penalty = torch.tensor(0.0, device=images.device)
                sigma_penalty = torch.tensor(0.0, device=images.device)
                sigma_mean = torch.tensor(0.0, device=images.device)
                sigma_min = torch.tensor(0.0, device=images.device)
                diffusion_loss = loss
                noise_loss = loss

        running["loss"] += loss.item()
        running["recon_loss"] += recon_loss.item()
        running["boundary_loss"] += boundary_loss.item()
        running["rank_penalty"] += rank_penalty.item()
        running["sigma_penalty"] += sigma_penalty.item()
        running["sigma_mean"] += sigma_mean.item()
        running["sigma_min"] += sigma_min.item()
        running["diffusion_loss"] += diffusion_loss.item()
        running["noise_loss"] += noise_loss.item()
        n_batches += 1

    metrics = {key: value / max(n_batches, 1) for key, value in running.items()}
    return metrics, None


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
    criterion = None
    consistency_weight = 0.0
    consistency_k_steps = 0
    num_classes = None

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
