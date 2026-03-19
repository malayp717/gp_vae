"""Training and validation loops for diffusion-family models."""

from __future__ import annotations

from typing import Any

import torch
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from vae.losses import VAELoss
from vae.models import get_kl_override
from vae.training.beta_schedule import BetaScheduler

from .evaluation import evaluate_loader


def train_one_epoch(
    model: torch.nn.Module,
    model_type: str,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    beta_scheduler: BetaScheduler,
    epoch: int,
    device: torch.device,
    scaler: GradScaler | None,
    grad_clip: float | None,
    use_amp: bool,
    criterion: VAELoss | None,
    consistency_weight: float,
    consistency_k_steps: int,
    ema_decay: float,
    disc_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float], float]:
    """Run a single diffusion training epoch and return aggregate metrics."""
    model.train()
    use_reconstruction_loss = model_type == "sccd"
    if criterion is not None and criterion.has_discriminator and use_reconstruction_loss:
        criterion.discriminator.train()  # type: ignore[union-attr]

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
    steps_per_epoch = len(loader)
    beta = 0.0

    pbar = tqdm(enumerate(loader), total=steps_per_epoch, desc=f"Epoch {epoch + 1:>3d} [train]", leave=False)
    for step, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        beta = beta_scheduler(epoch, step, steps_per_epoch) if use_reconstruction_loss else 0.0
        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=use_amp):
            if model_type == "sccd":
                assert criterion is not None
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
                vae_loss = torch.tensor(0.0, device=images.device)
                comps = {
                    "recon": torch.tensor(0.0, device=images.device),
                    "kl": torch.tensor(0.0, device=images.device),
                    "lpips": torch.tensor(0.0, device=images.device),
                    "adv_g": torch.tensor(0.0, device=images.device),
                }
                consistency_loss = torch.tensor(0.0, device=images.device)
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

        d_loss_val = 0.0
        if disc_optimizer is not None and out is not None:
            disc_optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=use_amp):
                d_loss = criterion.discriminator_loss(out["recon"].detach(), images)
            if scaler is not None:
                scaler.scale(d_loss).backward()
                scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(criterion.discriminator_params(), grad_clip or 1.0)
                scaler.step(disc_optimizer)
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(criterion.discriminator_params(), grad_clip or 1.0)
                disc_optimizer.step()
            d_loss_val = d_loss.item()

        if scaler is not None:
            scaler.update()
        if hasattr(model, "update_ema") and model_type == "sccd":
            model.update_ema(decay=ema_decay)

        running["loss"] += loss.item()
        running["recon_loss"] += comps["recon"].item()
        running["kl_loss"] += comps["kl"].item()
        running["diffusion_loss"] += diffusion_loss.item()
        running["consistency_loss"] += consistency_loss.item()
        running["noise_loss"] += noise_loss.item()
        running["lpips_loss"] += comps["lpips"].item()
        running["adv_g_loss"] += comps["adv_g"].item()
        running["d_loss"] += d_loss_val
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            recon=f"{comps['recon'].item():.3f}",
            kl=f"{comps['kl'].item():.3f}",
            diff=f"{diffusion_loss.item():.3f}",
            adv_g=f"{comps['adv_g'].item():.3f}",
            d=f"{d_loss_val:.3f}",
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
    criterion: VAELoss | None,
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
        return_kl_per_dim=True,
    )
