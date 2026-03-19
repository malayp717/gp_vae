"""Model construction helpers."""

from __future__ import annotations

from typing import Any

import torch

from .ddpm import DDPMModel
from .gp_vae import GeneralizedPosteriorVAE
from .sccd import SCCDModel
from .vae import VAE

def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Construct a supported model from config."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "vae")

    if model_type not in ("vae", "gp_vae", "sccd", "ddpm"):
        raise ValueError(f"Unknown model type: {model_type!r}. Supported: 'vae', 'gp_vae', 'sccd', 'ddpm'.")

    if model_type == "gp_vae":
        return GeneralizedPosteriorVAE(
            image_size=data_cfg.get("image_size", 32),
            patch_div=model_cfg.get("patch_div", 2),
            in_channels=data_cfg.get("in_channels", 3),
            latent_dim=model_cfg.get("latent_dim_per_patch", 128),
            encoder_channels=model_cfg.get("patch_encoder_channels", [32, 64, 128, 256]),
            decoder_channels=model_cfg.get("patch_decoder_channels", [256, 128, 64, 32]),
            covariance_rank=model_cfg.get("covariance_rank", 8),
            transformer_dim=model_cfg.get("transformer_dim", 256),
            transformer_heads=model_cfg.get("transformer_heads", 8),
            transformer_layers=model_cfg.get("transformer_layers", 4),
            transformer_dropout=model_cfg.get("transformer_dropout", 0.1),
            use_batch_norm=model_cfg.get("use_batch_norm", True),
        )

    elif model_type == "sccd":
        return SCCDModel.from_config(config)

    elif model_type == "ddpm":
        return DDPMModel.from_config(config)
    
    return VAE(
        in_channels=data_cfg.get("in_channels", 3),
        latent_dim=model_cfg.get("latent_dim", 256),
        encoder_channels=model_cfg.get("encoder_channels", [32, 64, 128, 128, 256]),
        decoder_channels=model_cfg.get("decoder_channels", [256, 128, 128, 64, 32]),
        image_size=data_cfg.get("image_size", 32),
        dropout=model_cfg.get("dropout", 0.1),
        use_batch_norm=model_cfg.get("use_batch_norm", True),
    )

