"""Training and validation loops."""

from __future__ import annotations

from typing import Any

import torch
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from vae.evaluation.runner import evaluate_loader
from vae.losses import VAELoss
from vae.models import get_kl_override

from .beta_schedule import BetaScheduler


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    beta_scheduler: BetaScheduler,
    epoch: int,
    device: torch.device,
    scaler: GradScaler | None,
    grad_clip: float | None,
    use_amp: bool,
    criterion: VAELoss,
    disc_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float], float]:
    """Run a single training epoch and return aggregate metrics."""
    model.train()
    if criterion.has_discriminator:
        criterion.discriminator.train()  # type: ignore[union-attr]

    running = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "lpips_loss": 0.0,
        "adv_g_loss": 0.0,
        "d_loss": 0.0,
    }
    n_batches = 0
    steps_per_epoch = len(loader)
    beta = 0.0

    pbar = tqdm(enumerate(loader), total=steps_per_epoch, desc=f"Epoch {epoch + 1:>3d} [train]", leave=False)
    for step, (images, _) in pbar:
        images = images.to(device, non_blocking=True)
        beta = beta_scheduler(epoch, step, steps_per_epoch)
        optimizer.zero_grad(set_to_none=True)

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
        if disc_optimizer is not None:
            disc_optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=use_amp):
                d_loss = criterion.discriminator_loss(recon.detach(), images)
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

        running["loss"] += loss.item()
        running["recon_loss"] += comps["recon"].item()
        running["kl_loss"] += comps["kl"].item()
        running["lpips_loss"] += comps["lpips"].item()
        running["adv_g_loss"] += comps["adv_g"].item()
        running["d_loss"] += d_loss_val
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            recon=f"{comps['recon'].item():.3f}",
            kl=f"{comps['kl'].item():.3f}",
            adv_g=f"{comps['adv_g'].item():.3f}",
            d=f"{d_loss_val:.3f}",
            beta=f"{beta:.4f}",
        )

    metrics = {key: value / max(n_batches, 1) for key, value in running.items()}
    return metrics, beta


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    beta: float,
    use_amp: bool,
    criterion: VAELoss,
) -> tuple[dict[str, float], torch.Tensor | None]:
    """Run a shared validation pass."""
    return evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        beta=beta,
        use_amp=use_amp,
        criterion=criterion,
        return_kl_per_dim=True,
    )

