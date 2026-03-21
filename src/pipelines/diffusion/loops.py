"""Training and validation loops for diffusion-family models."""

from __future__ import annotations

from typing import Any

import torch
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from src.training.beta_schedule import BetaScheduler

from .evaluation import evaluate_loader


def train_one_epoch(
    model: torch.nn.Module,
    model_type: str,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    beta_scheduler: BetaScheduler | None,
    epoch: int,
    device: torch.device,
    scaler: GradScaler | None,
    grad_clip: float | None,
    use_amp: bool,
    criterion: torch.nn.Module | None,
    consistency_weight: float,
    consistency_k_steps: int,
    ema_decay: float,
    disc_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float], float]:
    """Run a single diffusion training epoch and return aggregate metrics."""
    model.train()
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
    steps_per_epoch = len(loader)
    beta = 0.0

    pbar = tqdm(enumerate(loader), total=steps_per_epoch, desc=f"Epoch {epoch + 1:>3d} [train]", leave=False)
    for step, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        beta = 0.0
        optimizer.zero_grad(set_to_none=True)

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

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scaler is not None:
            scaler.update()

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

        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            recon=f"{recon_loss.item():.3f}",
            bnd=f"{boundary_loss.item():.3f}",
            rank=f"{rank_penalty.item():.3f}",
            diff=f"{diffusion_loss.item():.3f}",
            beta=f"{beta:.4f}",
        )

    metrics = {key: value / max(n_batches, 1) for key, value in running.items()}
    return metrics, beta


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    model_type: str,
    loader: Any,
    device: torch.device,
    beta: float,
    use_amp: bool,
    criterion: torch.nn.Module | None,
    consistency_weight: float,
    consistency_k_steps: int,
    ) -> tuple[dict[str, float], torch.Tensor | None]:
    """Run a shared validation pass for diffusion-family models."""
    return evaluate_loader(
        model=model,
        model_type=model_type,
        loader=loader,
        device=device,
        beta=beta,
        use_amp=use_amp,
        criterion=criterion,
        consistency_weight=consistency_weight,
        consistency_k_steps=consistency_k_steps,
        return_kl_per_dim=False,
    )
