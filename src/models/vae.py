from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn


# -------------------------
# Normalization helper
# -------------------------
def norm_layer(channels: int) -> nn.Module:
    return nn.GroupNorm(num_groups=8, num_channels=channels)


# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        latent_dim: int,
        image_size: int,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.image_size = image_size

        layers = []
        channels_in = in_channels
        current_size = image_size

        # Build conv stack and track spatial size
        for channels_out in hidden_channels:
            layers.append(
                nn.Conv2d(
                    channels_in,
                    channels_out,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            layers.append(norm_layer(channels_out))
            layers.append(nn.ReLU(inplace=True))

            channels_in = channels_out
            current_size = current_size // 2

        self.conv_stack = nn.Sequential(*layers)

        # Final spatial resolution inferred dynamically
        self.final_size = current_size
        self.flatten_dim = hidden_channels[-1] * current_size * current_size

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_stack(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        init_size: int,
    ) -> None:
        super().__init__()

        self.init_channels = hidden_channels[0]
        self.init_size = init_size

        self.fc = nn.Linear(
            latent_dim,
            self.init_channels * self.init_size * self.init_size,
        )

        layers = []
        channels_in = hidden_channels[0]

        for channels_out in hidden_channels[1:]:
            layers.append(
                nn.ConvTranspose2d(
                    channels_in,
                    channels_out,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            layers.append(norm_layer(channels_out))
            layers.append(nn.ReLU(inplace=True))
            channels_in = channels_out

        # Final layer
        layers.append(
            nn.ConvTranspose2d(
                channels_in,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

        layers.append(nn.Sigmoid())

        self.deconv_stack = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.init_channels, self.init_size, self.init_size)
        return self.deconv_stack(h)


# -------------------------
# VAE
# -------------------------
class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        image_size: int = 32,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=encoder_channels,
            latent_dim=latent_dim,
            image_size=image_size,
        )

        # Decoder uses inferred spatial size from encoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_channels=decoder_channels,
            out_channels=in_channels,
            init_size=self.encoder.final_size,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        # Fourth return must be z: training/eval loops unpack (recon, mu, log_var, _).
        return recon, mu, logvar, z

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)