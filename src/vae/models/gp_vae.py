"""Generalized posterior VAE with patch latents and structured covariance."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.losses.kl import low_rank_kl_per_dim


class LatentRefinement(nn.Module):
    def __init__(self, latent_dim: int, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.transformer(z)


def make_cnn(in_channels: int, channels: Sequence[int], use_bn: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_channels
    for channel in channels:
        layers.append(nn.Conv2d(current, channel, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(channel))
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(2))
        current = channel
    return nn.Sequential(*layers)


class PatchTokenDecoder(nn.Module):
    """Decode a single token into a patch image."""

    def __init__(
        self,
        token_dim: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels must be non-empty")
        self.patch_size = int(patch_size)
        n_ups = max(len(hidden_channels) - 1, 0)
        self._init_channels = int(hidden_channels[0])
        self._init_size = max(1, self.patch_size // (2 ** max(n_ups, 1)))
        self.fc = nn.Linear(token_dim, self._init_channels * self._init_size * self._init_size)

        layers: list[nn.Module] = []
        channels_in = self._init_channels
        for channels_out in hidden_channels[1:]:
            layers.append(nn.ConvTranspose2d(channels_in, channels_out, 4, stride=2, padding=1))
            layers.append(nn.GELU())
            channels_in = channels_out
        layers.append(nn.Conv2d(channels_in, out_channels, 3, padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(token)
        hidden = hidden.view(-1, self._init_channels, self._init_size, self._init_size)
        hidden = self.deconv(hidden)
        if hidden.shape[-1] != self.patch_size:
            hidden = F.interpolate(hidden, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)
        return hidden


def _build_seam_mask(image_size: int, patch_size: int, width: int = 2) -> torch.Tensor:
    mask = torch.zeros(1, 1, image_size, image_size, dtype=torch.float32)
    grid = image_size // patch_size
    width = max(1, int(width))
    for idx in range(1, grid):
        pos = idx * patch_size
        y0, y1 = max(pos - width, 0), min(pos + width, image_size)
        x0, x1 = max(pos - width, 0), min(pos + width, image_size)
        mask[:, :, y0:y1, :] = 1.0
        mask[:, :, :, x0:x1] = 1.0
    return mask


class GeneralizedPosteriorVAE(nn.Module):
    """Patch-based VAE with low-rank posterior covariance."""

    def __init__(
        self,
        image_size: int = 224,
        patch_div: int = 4,
        in_channels: int = 3,
        latent_dim: int = 32,
        covariance_rank: int = 8,
        encoder_channels: Sequence[int] = (32, 64, 128),
        decoder_channels: Sequence[int] = (128, 64, 32),
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        transformer_dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_div = patch_div
        self.patch_size = image_size // patch_div
        self.num_patches = patch_div ** 2
        self.latent_dim = latent_dim
        self.cov_rank = covariance_rank

        self.patch_cnn = make_cnn(in_channels, encoder_channels, use_bn=use_batch_norm)
        feat_dim = encoder_channels[-1]
        self.enc_proj = nn.Linear(feat_dim, transformer_dim)
        self.encoder_pos = nn.Parameter(torch.randn(1, self.num_patches, transformer_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            transformer_dim,
            transformer_heads,
            transformer_dim * 4,
            dropout=transformer_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer,
            transformer_layers,
            enable_nested_tensor=False,
        )
        self.fc_mu = nn.Linear(transformer_dim, latent_dim)
        self.fc_log_sigma = nn.Linear(transformer_dim, latent_dim)
        self.fc_V = nn.Linear(transformer_dim, latent_dim * covariance_rank)

        self.global_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.latent_pos = nn.Parameter(torch.randn(1, self.num_patches + 1, latent_dim) * 0.02)

        latent_heads = min(int(transformer_heads), int(latent_dim))
        while latent_heads > 1 and (latent_dim % latent_heads) != 0:
            latent_heads -= 1
        self.latent_refine = LatentRefinement(
            latent_dim,
            heads=max(1, latent_heads),
            layers=2,
            dropout=transformer_dropout,
        )

        self.dec_proj = nn.Linear(latent_dim, transformer_dim)
        dec_layer = nn.TransformerEncoderLayer(
            transformer_dim,
            transformer_heads,
            transformer_dim * 4,
            dropout=transformer_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerEncoder(
            dec_layer,
            transformer_layers,
            enable_nested_tensor=False,
        )
        self.patch_decoder = PatchTokenDecoder(
            token_dim=transformer_dim,
            hidden_channels=decoder_channels,
            out_channels=in_channels,
            patch_size=self.patch_size,
        )
        self.seam_refiner = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )
        self.register_buffer(
            "seam_mask",
            _build_seam_mask(self.image_size, self.patch_size, width=2),
            persistent=False,
        )
        self._cached_kl: torch.Tensor | None = None
        self._cached_kl_per_dim: torch.Tensor | None = None

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        patch = self.patch_size
        patches = x.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        return patches.reshape(batch, self.num_patches, channels, patch, patch)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch, _, channels, patch, _ = patches.shape
        grid = self.patch_div
        patches = patches.reshape(batch, grid, grid, channels, patch, patch)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(batch, channels, grid * patch, grid * patch)

    def encode_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        patches = self.patchify(x)
        flat = patches.reshape(batch * self.num_patches, patches.size(2), self.patch_size, self.patch_size)
        feats = self.patch_cnn(flat).mean([2, 3])
        feats = feats.view(batch, self.num_patches, -1)

        tokens = self.enc_proj(feats) + self.encoder_pos
        ctx = self.transformer_encoder(tokens)
        mu = self.fc_mu(ctx)
        log_sigma = self.fc_log_sigma(ctx)
        V = self.fc_V(ctx).view(batch, self.num_patches, self.latent_dim, self.cov_rank)
        return mu, log_sigma, V

    def encode_latent_mean(self, x: torch.Tensor) -> torch.Tensor:
        mu, _, _ = self.encode_distribution(x)
        return mu

    def reparam(self, mu: torch.Tensor, log_sigma: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        eps1 = torch.randn_like(mu)
        eps2 = torch.randn(*mu.shape[:-1], self.cov_rank, device=mu.device)
        return mu + sigma * eps1 + (V @ eps2.unsqueeze(-1)).squeeze(-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.size(0)
        tokens = self.dec_proj(z)
        tokens = self.transformer_decoder(tokens)
        tokens_flat = tokens.reshape(batch * self.num_patches, -1)
        patches = self.patch_decoder(tokens_flat)
        patches = patches.view(batch, self.num_patches, patches.size(1), self.patch_size, self.patch_size)
        image = self.unpatchify(patches)
        delta = self.seam_refiner(image)
        image = image + self.seam_mask.to(dtype=image.dtype, device=image.device) * delta
        return torch.sigmoid(image)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        mu, log_sigma, V = self.encode_distribution(x)
        z = self.reparam(mu, log_sigma, V)
        z = z / (z.std(dim=-1, keepdim=True) + 1e-6)
        global_token = self.global_token.expand(batch, -1, -1)
        z = torch.cat([global_token, z], dim=1)
        z = z + self.latent_pos
        z = self.latent_refine(z)
        z = z[:, 1:]
        recon = self.decode(z)

        kl_tokens, kl_per_dim_tokens = low_rank_kl_per_dim(
            mu.reshape(-1, self.latent_dim),
            log_sigma.reshape(-1, self.latent_dim),
            V.reshape(-1, self.latent_dim, self.cov_rank),
        )
        self._cached_kl = kl_tokens * self.num_patches
        self._cached_kl_per_dim = kl_per_dim_tokens * self.num_patches
        log_var = 2.0 * log_sigma
        return recon, mu, log_var, z

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: torch.device,
        temperature: float = 1.0,
        truncation: float | None = None,
    ) -> torch.Tensor:
        z = torch.randn(n_samples, self.num_patches, self.latent_dim, device=device) * float(temperature)
        if truncation is not None:
            z = z.clamp(-float(truncation), float(truncation))
        z = z / (z.std(dim=-1, keepdim=True) + 1e-6)
        global_token = self.global_token.expand(n_samples, -1, -1)
        z = torch.cat([global_token, z], dim=1)
        z = z + self.latent_pos
        z = self.latent_refine(z)
        z = z[:, 1:]
        return self.decode(z)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode_distribution(x)

    def get_kl_override(self) -> torch.Tensor | None:
        return self._cached_kl

    def get_kl_per_dim(self) -> torch.Tensor | None:
        return self._cached_kl_per_dim

