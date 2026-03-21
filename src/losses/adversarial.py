"""Adversarial loss components."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """Markovian (PatchGAN) discriminator."""

    @staticmethod
    def _sn_conv2d(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
    ) -> nn.Conv2d:
        return spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            self._sn_conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for _ in range(1, n_layers):
            next_channels = min(channels * 2, 512)
            layers += [
                self._sn_conv2d(channels, next_channels, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = next_channels

        next_channels = min(channels * 2, 512)
        layers += [
            self._sn_conv2d(channels, next_channels, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(next_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers.append(self._sn_conv2d(next_channels, 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

