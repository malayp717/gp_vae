"""Convolutional variational autoencoder."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional encoder producing mean and log-variance vectors."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        latent_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels_in = in_channels
        for channels_out in hidden_channels:
            layers.append(nn.Conv2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(channels_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            channels_in = channels_out
        self.conv_stack = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(hidden_channels[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.conv_stack(x)
        pooled = self.adaptive_pool(hidden).flatten(start_dim=1)
        return self.fc_mu(pooled), self.fc_log_var(pooled)


class Decoder(nn.Module):
    """Convolutional decoder that maps latent codes back to images."""

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        image_size: int,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self._init_channels = hidden_channels[0]
        n_down = len(hidden_channels)
        self._init_size = max(1, image_size // (2 ** n_down))
        self.fc = nn.Linear(latent_dim, hidden_channels[0] * self._init_size * self._init_size)

        layers: list[nn.Module] = []
        channels_in = hidden_channels[0]
        for channels_out in hidden_channels[1:]:
            layers.append(nn.ConvTranspose2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(channels_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels_in = channels_out

        layers.append(nn.ConvTranspose2d(channels_in, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())
        self.deconv_stack = nn.Sequential(*layers)
        self._image_size = image_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(z)
        hidden = hidden.view(-1, self._init_channels, self._init_size, self._init_size)
        hidden = self.deconv_stack(hidden)
        return F.interpolate(hidden, size=self._image_size, mode="bilinear", align_corners=False)


class VAE(nn.Module):
    """Beta-VAE with convolutional encoder/decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 512,
        encoder_channels: Sequence[int] = (32, 64, 128, 256, 512),
        decoder_channels: Sequence[int] = (512, 256, 128, 64, 32),
        image_size: int = 224,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=list(encoder_channels),
            latent_dim=latent_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_channels=list(decoder_channels),
            out_channels=in_channels,
            image_size=image_size,
            use_batch_norm=use_batch_norm,
        )
        self._cached_kl: torch.Tensor | None = None
        self._cached_kl_per_dim: torch.Tensor | None = None
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def encode_latent_mean(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_distribution(x)
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode_distribution(x)
        z = self.reparameterise(mu, log_var)
        reconstruction = self.decoder(z)
        self._cached_kl = None
        self._cached_kl_per_dim = None
        return reconstruction, mu, log_var, z

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_distribution(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def get_kl_override(self) -> torch.Tensor | None:
        return self._cached_kl

    def get_kl_per_dim(self) -> torch.Tensor | None:
        return self._cached_kl_per_dim

