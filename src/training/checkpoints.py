"""Checkpoint save/load helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from src.models import build_model
from src.runtime.device import get_device

logger = logging.getLogger(__name__)


def model_checkpoint_dir(checkpoint_dir: Path, model_type: str) -> Path:
    """Return the subdirectory that stores checkpoints for one model type."""
    return checkpoint_dir / model_type


def periodic_checkpoint_path(checkpoint_dir: Path, model_type: str, epoch: int) -> Path:
    """Return the periodic checkpoint path for a model and epoch index."""
    model_dir = model_checkpoint_dir(checkpoint_dir, model_type)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{model_type}_epoch_{epoch + 1:04d}.pt"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_loss: float,
    config: dict[str, Any],
    scaler: Any | None = None,
    criterion: torch.nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    data: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "val_loss": val_loss,
        "config": config,
    }
    if criterion is not None:
        data["criterion_state_dict"] = criterion.state_dict()
    if disc_optimizer is not None:
        data["disc_optimizer_state_dict"] = disc_optimizer.state_dict()
    torch.save(data, path)


def load_training_state(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: torch.device | None = None,
    criterion: torch.nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float]:
    if device is None:
        device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if criterion and ckpt.get("criterion_state_dict"):
        criterion.load_state_dict(ckpt["criterion_state_dict"])
    if disc_optimizer and ckpt.get("disc_optimizer_state_dict"):
        disc_optimizer.load_state_dict(ckpt["disc_optimizer_state_dict"])

    start_epoch = ckpt.get("epoch", -1) + 1
    best_val_loss = ckpt.get("val_loss", float("inf"))
    logger.info(
        "Resumed from %s - continuing at epoch %d (best_val_loss=%.4f)",
        checkpoint_path,
        start_epoch,
        best_val_loss,
    )
    return start_epoch, best_val_loss


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if device is None:
        device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: dict[str, Any] = ckpt["config"]
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(
        "Loaded checkpoint from %s (epoch %d, val_loss=%.4f)",
        checkpoint_path,
        ckpt.get("epoch", -1),
        ckpt.get("val_loss", float("nan")),
    )
    return model, config


def latest_checkpoint_path(
    checkpoint_dir: Path,
    model_type: str,
) -> Path:
    """Return the latest periodic checkpoint for the model."""
    model_dir = model_checkpoint_dir(checkpoint_dir, model_type)
    checkpoints = sorted(model_dir.glob(f"{model_type}_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No periodic checkpoints found for model '{model_type}' in {model_dir}"
        )
    return checkpoints[-1]

