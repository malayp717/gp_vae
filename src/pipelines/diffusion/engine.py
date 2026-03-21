"""Top-level training engine for diffusion-family models."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler
from torch.optim import AdamW

from src.config.loader import load_config
from src.data import get_cifar10_dataloaders
from src.models import build_model
from src.runtime import get_device, get_run_output_dir, seed_everything
from src.training.checkpoints import load_training_state, periodic_checkpoint_path, save_checkpoint
from src.training.lr_schedulers import build_cosine_warmup_scheduler

from .evaluation import compute_image_metrics
from .generation import save_validation_images
from .loops import train_one_epoch, validate_one_epoch
from .reporting import append_train_stats, append_val_stats, divider_for, fmt_row, header_for

logger = logging.getLogger(__name__)


def train(
    config_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Run full diffusion-family training and return the latest checkpoint path."""
    config = load_config(config_path, overrides=config_overrides)
    model_cfg = config["model"]
    train_cfg = config["training"]
    opt_cfg = config.get("optimizer", {})
    sched_cfg = config.get("scheduler", {})
    paths_cfg = config["paths"]
    log_cfg = config.get("logging", {})
    loss_cfg = config.get("loss", {})

    seed_everything(train_cfg.get("seed", 42))
    device = get_device()
    logger.info("Device: %s", device)

    model_type = model_cfg.get("type", "sccd")
    logger.info("Model type: %s", model_type)

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "./checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = get_run_output_dir(config)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = logs_dir / "train_stats.txt"
    val_log_path = logs_dir / "val_stats.txt"

    train_loader, val_loader, _ = get_cifar10_dataloaders(config)
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    model = build_model(config).to(device)
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")
    if train_cfg.get("compile_model", False):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=opt_cfg.get("eps", 1e-8),
    )
    epochs = train_cfg.get("epochs", 200)
    lr_scheduler = build_cosine_warmup_scheduler(
        optimizer,
        warmup_epochs=sched_cfg.get("warmup_epochs", 5),
        total_epochs=epochs,
        min_lr=sched_cfg.get("min_lr", 1e-6),
        base_lr=train_cfg.get("learning_rate", 3e-4),
    )
    if model_type == "ddpm":
        sample_steps = int(model_cfg.get("ddpm_sample_steps", model_cfg.get("diffusion_T", 1000)))
        criterion = None
        logger.info(
            "Loss config: ddpm_denoising | beta_schedule=%s | T=%d | sample_steps=%d",
            model_cfg.get("ddpm_beta_schedule", "linear"),
            int(model_cfg.get("diffusion_T", 1000)),
            sample_steps,
        )
    else:
        sample_steps = int(model_cfg.get("sample_steps", 1))
        criterion = None
        logger.info(
            "Loss config: scpdt_native | lambda_recon=%.3f | lambda_boundary=%.3f | lambda_rank=%.3f | steps=%d | mode=%s",
            float(loss_cfg.get("sccd_lambda_recon", 1.0)),
            float(loss_cfg.get("sccd_lambda_boundary", 0.1)),
            float(loss_cfg.get("sccd_lambda_rank", 0.01)),
            sample_steps,
            str(model_cfg.get("sccd_sampler_mode", "hybrid")),
        )

    disc_optimizer: torch.optim.Optimizer | None = None
    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from is None:
        resume_from = paths_cfg.get("resume_from")
    if resume_from and Path(resume_from).exists():
        start_epoch, best_val_loss = load_training_state(
            resume_from,
            model,
            optimizer,
            lr_scheduler,
            scaler,
            device,
            criterion=criterion,
            disc_optimizer=disc_optimizer,
        )

    grad_clip = train_cfg.get("gradient_clip_norm")
    save_interval = log_cfg.get("save_interval", 5)
    eval_interval = log_cfg.get("eval_interval", 5)
    num_fid_samples = log_cfg.get("num_fid_samples", 1024)
    patience = train_cfg.get("early_stopping_patience", 20)
    num_classes = None

    epochs_without_improvement = 0
    latest_ckpt_path: Path | None = None

    divider = divider_for(model_type)
    print(f"\n{divider}")
    print(header_for(model_type))
    print(divider)
    t_start = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        train_metrics, last_beta = train_one_epoch(
            model,
            model_type,
            train_loader,
            optimizer,
            None,
            epoch,
            device,
            scaler,
            grad_clip,
            use_amp,
            criterion=criterion,
            consistency_weight=0.0,
            consistency_k_steps=0,
            ema_decay=0.0,
            disc_optimizer=disc_optimizer,
        )
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            fmt_row(
                model_type,
                epoch,
                epochs,
                "train",
                train_metrics["loss"],
                train_metrics["recon_loss"],
                boundary_loss=train_metrics["boundary_loss"],
                rank_penalty=train_metrics["rank_penalty"],
                sigma_mean=train_metrics.get("sigma_mean", 0.0),
                sigma_min=train_metrics.get("sigma_min", 0.0),
                diffusion_loss=train_metrics["diffusion_loss"],
                lr=current_lr,
            )
        )
        append_train_stats(train_log_path, model_type, epoch, epochs, train_metrics)

        if (epoch + 1) % eval_interval == 0:
            val_metrics, kl_per_dim = validate_one_epoch(
                model,
                model_type,
                val_loader,
                device,
                0.0,
                use_amp,
                criterion=criterion,
                consistency_weight=0.0,
                consistency_k_steps=0,
            )
            img_metrics = compute_image_metrics(
                model,
                model_type,
                val_loader,
                device,
                num_classes=num_classes,
                num_fid_samples=num_fid_samples,
                sample_steps=sample_steps,
            )

            save_validation_images(
                model,
                val_loader,
                device,
                epoch,
                model_type,
                run_output_dir,
                use_amp,
                num_classes=0,
                steps=sample_steps,
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                fmt_row(
                    model_type,
                    epoch,
                    epochs,
                    "val",
                    val_metrics["loss"],
                    val_metrics["recon_loss"],
                    boundary_loss=val_metrics["boundary_loss"],
                    rank_penalty=val_metrics["rank_penalty"],
                    sigma_mean=val_metrics.get("sigma_mean", 0.0),
                    sigma_min=val_metrics.get("sigma_min", 0.0),
                    diffusion_loss=val_metrics["diffusion_loss"],
                    lr=current_lr,
                    fid=img_metrics["fid"],
                    ssim=img_metrics["ssim"],
                    psnr=img_metrics["psnr"],
                )
            )
            append_val_stats(
                val_log_path,
                model_type,
                epoch,
                epochs,
                val_metrics,
                fid=img_metrics["fid"],
                ssim=img_metrics["ssim"],
                psnr=img_metrics["psnr"],
            )

            if patience and epochs_without_improvement >= patience:
                print(divider)
                print(f"  Early stopping - no improvement for {patience} epochs.")
                break

        if save_interval and (epoch + 1) % save_interval == 0:
            ckpt_path = periodic_checkpoint_path(checkpoint_dir, model_type, epoch)
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                lr_scheduler,
                epoch,
                best_val_loss,
                config,
                scaler,
                criterion=criterion,
                disc_optimizer=disc_optimizer,
            )
            latest_ckpt_path = ckpt_path

    elapsed = time.perf_counter() - t_start
    if latest_ckpt_path is None:
        latest_ckpt_path = periodic_checkpoint_path(checkpoint_dir, model_type, epoch)
        save_checkpoint(
            latest_ckpt_path,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            best_val_loss,
            config,
            scaler,
            criterion=criterion,
            disc_optimizer=disc_optimizer,
        )
    print(divider)
    print(f"  Training complete in {elapsed:.1f}s | Best val loss: {best_val_loss:.4f}")
    print(f"  Latest checkpoint: {latest_ckpt_path}\n")
    return latest_ckpt_path
