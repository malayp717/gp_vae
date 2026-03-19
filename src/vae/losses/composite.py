"""Composite training loss for VAE models."""

from __future__ import annotations

import warnings
from typing import Any, Iterator

import lpips as _lpips_lib
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adversarial import PatchDiscriminator


class VAELoss(nn.Module):
    """Composite VAE loss: MSE + beta-KL + optional LPIPS + optional adversarial."""

    def __init__(
        self,
        recon_weight: float = 1.0,
        lpips_enabled: bool = False,
        lpips_weight: float = 0.0,
        lpips_net: str = "vgg",
        adv_enabled: bool = False,
        adv_weight: float = 0.0,
        adv_start_epoch: int = 0,
        disc_in_channels: int = 3,
        disc_channels: int = 64,
        disc_layers: int = 3,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.recon_weight = recon_weight
        self.lpips_enabled = lpips_enabled
        self.lpips_weight = lpips_weight if lpips_enabled else 0.0
        self._lpips_fn: _lpips_lib.LPIPS | None = None
        if lpips_enabled and lpips_weight > 0:
            with warnings.catch_warnings():
                # LPIPS still constructs torchvision backbones with deprecated args internally.
                warnings.filterwarnings(
                    "ignore",
                    message=r"The parameter 'pretrained' is deprecated since 0\.13.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
                    category=UserWarning,
                )
                self._lpips_fn = _lpips_lib.LPIPS(net=lpips_net).to(device)
            self._lpips_fn.eval()
            for param in self._lpips_fn.parameters():
                param.requires_grad = False

        self.adv_enabled = adv_enabled
        self.adv_weight = adv_weight if adv_enabled else 0.0
        self.adv_start_epoch = adv_start_epoch
        self._current_epoch = 0
        self._discriminator: PatchDiscriminator | None = None
        if adv_enabled and adv_weight > 0:
            self._discriminator = PatchDiscriminator(
                in_channels=disc_in_channels,
                base_channels=disc_channels,
                n_layers=disc_layers,
            ).to(device)

    @property
    def has_discriminator(self) -> bool:
        return self._discriminator is not None

    @property
    def discriminator(self) -> PatchDiscriminator | None:
        return self._discriminator

    def discriminator_params(self) -> Iterator[torch.nn.Parameter]:
        if self._discriminator is not None:
            yield from self._discriminator.parameters()

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    @property
    def effective_adv_weight(self) -> float:
        if self._current_epoch < self.adv_start_epoch:
            return 0.0
        return self.adv_weight

    @staticmethod
    def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, target, reduction="sum") / target.size(0)

    @staticmethod
    def kl_divergence(
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kl_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kl_override is not None:
            return kl_override
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)

    def perceptual_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._lpips_fn is None:
            return torch.tensor(0.0, device=recon.device)
        return self._lpips_fn(recon, target, normalize=True).mean()

    def generator_adv_loss(self, recon: torch.Tensor) -> torch.Tensor:
        if self._discriminator is None:
            return torch.tensor(0.0, device=recon.device)
        fake_logits = self._discriminator(recon)
        return -fake_logits.clamp(-50.0, 50.0).mean()

    def discriminator_loss(
        self,
        recon_detached: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        assert self._discriminator is not None, "No discriminator to train"
        real_logits = self._discriminator(real).clamp(-50.0, 50.0)
        fake_logits = self._discriminator(recon_detached).clamp(-50.0, 50.0)
        return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
        kl_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        recon_loss = self.reconstruction_loss(recon, target)
        kl_loss = self.kl_divergence(mu, log_var, kl_override)
        total = self.recon_weight * recon_loss + beta * kl_loss
        zero = torch.tensor(0.0, device=recon.device)

        if self._lpips_fn is not None and self.lpips_weight > 0:
            lpips_loss = self.perceptual_loss(recon, target)
            total = total + self.lpips_weight * lpips_loss
        else:
            lpips_loss = zero

        adv_weight = self.effective_adv_weight
        if self._discriminator is not None and adv_weight > 0:
            adv_g_loss = self.generator_adv_loss(recon)
            total = total + adv_weight * adv_g_loss
        else:
            adv_g_loss = zero

        return total, {
            "recon": recon_loss,
            "kl": kl_loss,
            "lpips": lpips_loss,
            "adv_g": adv_g_loss,
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> "VAELoss":
        loss_cfg = config.get("loss", {})
        data_cfg = config.get("data", {})
        return cls(
            recon_weight=loss_cfg.get("recon_weight", 1.0),
            lpips_enabled=loss_cfg.get("lpips_enabled", False),
            lpips_weight=loss_cfg.get("lpips_weight", 0.0),
            lpips_net=loss_cfg.get("lpips_net", "vgg"),
            adv_enabled=loss_cfg.get("adv_enabled", False),
            adv_weight=loss_cfg.get("adv_weight", 0.0),
            adv_start_epoch=loss_cfg.get("adv_start_epoch", 0),
            disc_in_channels=data_cfg.get("in_channels", 3),
            disc_channels=loss_cfg.get("disc_channels", 64),
            disc_layers=loss_cfg.get("disc_layers", 3),
            device=device,
        )

