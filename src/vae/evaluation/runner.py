"""Evaluation runners shared by training and standalone validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from vae.config.loader import load_config
from vae.data import CIFAR10_CLASSES, get_cifar10_dataloaders
from vae.losses import VAELoss, diagonal_kl_per_dim
from vae.models import get_kl_override, get_kl_per_dim
from vae.runtime import get_device
from vae.training.checkpoints import load_model_from_checkpoint

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    beta: float = 1.0,
    use_amp: bool = False,
    criterion: VAELoss | None = None,
    return_kl_per_dim: bool = False,
) -> tuple[dict[str, float], torch.Tensor | None]:
    """Compute aggregate metrics over a loader and optionally KL-per-dim."""
    model.eval()
    criterion = criterion or VAELoss()
    if criterion.has_discriminator:
        criterion.discriminator.eval()  # type: ignore[union-attr]

    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "lpips_loss": 0.0,
        "adv_g_loss": 0.0,
        "d_loss": 0.0,
    }
    n_batches = 0
    n_images = 0
    kl_per_dim_sum: torch.Tensor | None = None

    for images, _ in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp):
            recon, mu, log_var, _ = model(images)
            loss, comps = criterion(
                recon,
                images,
                mu,
                log_var,
                beta,
                kl_override=get_kl_override(model),
            )

        running["loss"] += loss.item()
        running["recon_loss"] += comps["recon"].item()
        running["kl_loss"] += comps["kl"].item()
        running["lpips_loss"] += comps["lpips"].item()
        running["adv_g_loss"] += comps["adv_g"].item()
        if criterion.has_discriminator:
            running["d_loss"] += criterion.discriminator_loss(recon, images).item()
        n_batches += 1
        n_images += images.size(0)

        if return_kl_per_dim:
            batch_kl_per_dim = get_kl_per_dim(model)
            if batch_kl_per_dim is None:
                batch_kl_per_dim = diagonal_kl_per_dim(mu, log_var)
            if kl_per_dim_sum is None:
                kl_per_dim_sum = torch.zeros_like(batch_kl_per_dim, dtype=torch.float32, device="cpu")
            kl_per_dim_sum += batch_kl_per_dim.detach().to("cpu", dtype=torch.float32) * images.size(0)

    metrics = {key: value / max(n_batches, 1) for key, value in running.items()}
    if not return_kl_per_dim or kl_per_dim_sum is None or n_images == 0:
        return metrics, None
    return metrics, kl_per_dim_sum / float(n_images)


@torch.no_grad()
def evaluate_per_class(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    beta: float = 1.0,
    use_amp: bool = False,
    criterion: VAELoss | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-class reconstruction and KL metrics."""
    model.eval()
    criterion = criterion or VAELoss()
    accum: dict[int, dict[str, float]] = {}

    for images, labels in tqdm(loader, desc="Per-class eval", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp):
            recon, mu, log_var, _ = model(images)

        for idx in range(images.size(0)):
            img = images[idx].unsqueeze(0)
            rec = recon[idx].unsqueeze(0)
            mu_i = mu[idx].unsqueeze(0)
            log_var_i = log_var[idx].unsqueeze(0)
            _, components = criterion(rec, img, mu_i, log_var_i, beta, kl_override=None)
            cls = labels[idx].item()
            if cls not in accum:
                accum[cls] = {"recon_loss": 0.0, "kl_loss": 0.0, "count": 0.0}
            accum[cls]["recon_loss"] += components["recon"].item()
            accum[cls]["kl_loss"] += components["kl"].item()
            accum[cls]["count"] += 1

    results: dict[str, dict[str, float]] = {}
    for cls_idx in sorted(accum):
        count = accum[cls_idx]["count"]
        name = CIFAR10_CLASSES[cls_idx] if cls_idx < len(CIFAR10_CLASSES) else str(cls_idx)
        results[name] = {
            "recon_loss": accum[cls_idx]["recon_loss"] / count,
            "kl_loss": accum[cls_idx]["kl_loss"] / count,
            "count": int(count),
        }
    return results


def run_validation(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, float]:
    """Load a checkpoint, run evaluation, and log aggregate metrics."""
    device = get_device()
    model, ckpt_config = load_model_from_checkpoint(checkpoint_path, device)
    config = load_config(config_path) if config_path else ckpt_config
    _, val_loader, test_loader = get_cifar10_dataloaders(config)
    loader = val_loader if split == "val" else test_loader
    use_amp = config.get("training", {}).get("mixed_precision", True) and device.type == "cuda"
    criterion = VAELoss.from_config(config, device)

    metrics, _ = evaluate_loader(model, loader, device, beta=1.0, use_amp=use_amp, criterion=criterion)
    logger.info("=== %s metrics ===", split.upper())
    for key, value in metrics.items():
        logger.info("  %-12s %.4f", key, value)

    per_class = evaluate_per_class(model, loader, device, beta=1.0, use_amp=use_amp, criterion=criterion)
    logger.info("--- per-class recon_loss ---")
    for cls_name, values in per_class.items():
        logger.info(
            "  %-12s recon=%.4f  kl=%.4f  (n=%d)",
            cls_name,
            values["recon_loss"],
            values["kl_loss"],
            values["count"],
        )
    return metrics

