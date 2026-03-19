"""Vanilla pixel-space DDPM with a timestep-conditioned U-Net denoiser."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_beta_schedule(
    schedule: str,
    timesteps: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, timesteps, device=device, dtype=torch.float32)
    if schedule == "cosine":
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, device=device, dtype=torch.float32) / timesteps
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999)
    raise ValueError(f"Unsupported DDPM beta schedule: {schedule!r}")


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetDenoiser(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_embed_dim = base_channels
        time_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(base_channels)
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        ch = base_channels
        self.down_levels = nn.ModuleList()
        skip_channels: list[int] = []
        for level_idx, multiplier in enumerate(channel_multipliers):
            out_ch = base_channels * int(multiplier)
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, time_dim, dropout=dropout))
                ch = out_ch
                skip_channels.append(ch)
            downsample = Downsample(ch) if level_idx < len(channel_multipliers) - 1 else None
            self.down_levels.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "downsample": downsample if downsample is not None else nn.Identity(),
                    }
                )
            )

        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout=dropout)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout=dropout)

        self.up_levels = nn.ModuleList()
        for level_idx, multiplier in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * int(multiplier)
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout=dropout))
                ch = out_ch
            upsample = Upsample(ch) if level_idx > 0 else None
            self.up_levels.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "upsample": upsample if upsample is not None else nn.Identity(),
                    }
                )
            )

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)
        h = self.input_conv(x)
        skips: list[torch.Tensor] = []

        for level_idx, level in enumerate(self.down_levels):
            for block in level["blocks"]:
                h = block(h, t_emb)
                skips.append(h)
            if level_idx < len(self.down_levels) - 1:
                h = level["downsample"](h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for level_idx, level in enumerate(self.up_levels):
            for block in level["blocks"]:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            if level_idx < len(self.up_levels) - 1:
                h = level["upsample"](h)

        return self.out_conv(self.out_act(self.out_norm(h)))


class DDPMModel(nn.Module):
    """Classic unconditional pixel-space DDPM."""

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        diffusion_T: int = 1000,
        beta_schedule: str = "linear",
        sample_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.channel_multipliers = tuple(int(v) for v in channel_multipliers)
        self.num_res_blocks = int(num_res_blocks)
        self.dropout = float(dropout)
        self.diffusion_T = int(diffusion_T)
        self.sample_steps = int(sample_steps)
        self.beta_schedule = beta_schedule

        self.denoiser = UNetDenoiser(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            channel_multipliers=self.channel_multipliers,
            num_res_blocks=self.num_res_blocks,
            dropout=self.dropout,
        )

        betas = _make_beta_schedule(beta_schedule, self.diffusion_T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas), persistent=False)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20), persistent=False)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x_t, timesteps)

    @staticmethod
    def _to_model_space(images: torch.Tensor) -> torch.Tensor:
        return images * 2.0 - 1.0

    @staticmethod
    def _to_image_space(images: torch.Tensor) -> torch.Tensor:
        return ((images + 1.0) / 2.0).clamp(0.0, 1.0)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def training_loss(self, images: torch.Tensor) -> torch.Tensor:
        x_start = self._to_model_space(images)
        timesteps = torch.randint(0, self.diffusion_T, (x_start.size(0),), device=x_start.device)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise=noise)
        pred_noise = self(noisy, timesteps)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: torch.device,
        steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = int(self.sample_steps if steps is None else steps)
        steps = max(1, min(steps, self.diffusion_T))

        if initial_noise is None:
            x = torch.randn(n_samples, self.in_channels, self.image_size, self.image_size, device=device)
        else:
            x = initial_noise.to(device)
            n_samples = x.size(0)

        if steps == self.diffusion_T:
            timesteps = list(range(self.diffusion_T - 1, -1, -1))
        else:
            timesteps = torch.linspace(self.diffusion_T - 1, 0, steps, device=device).long().tolist()

        for timestep in timesteps:
            t = torch.full((n_samples,), int(timestep), device=device, dtype=torch.long)
            pred_noise = self(x, t)
            beta_t = self.betas[t].view(-1, 1, 1, 1)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
            mean = sqrt_recip_alpha * (x - beta_t * pred_noise / sqrt_one_minus.clamp(min=1e-8))

            if int(timestep) > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[t].view(-1, 1, 1, 1)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean

        return self._to_image_space(x)

    @torch.no_grad()
    def sample_interpolations(
        self,
        n_pairs: int,
        n_steps: int,
        device: torch.device,
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for _ in range(n_pairs):
            noise_a = torch.randn(1, self.in_channels, self.image_size, self.image_size, device=device)
            noise_b = torch.randn(1, self.in_channels, self.image_size, self.image_size, device=device)
            alphas = torch.linspace(0.0, 1.0, n_steps, device=device)
            interp_noise = torch.cat([(1.0 - alpha) * noise_a + alpha * noise_b for alpha in alphas], dim=0)
            rows.append(self.sample(interp_noise.size(0), device, steps=sample_steps, initial_noise=interp_noise))
        return torch.cat(rows, dim=0)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DDPMModel":
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        return cls(
            image_size=data_cfg.get("image_size", 32),
            in_channels=data_cfg.get("in_channels", 3),
            base_channels=model_cfg.get("ddpm_base_channels", 64),
            channel_multipliers=model_cfg.get("ddpm_channel_multipliers", [1, 2, 4]),
            num_res_blocks=model_cfg.get("ddpm_num_res_blocks", 2),
            dropout=model_cfg.get("ddpm_dropout", 0.0),
            diffusion_T=model_cfg.get("diffusion_T", 1000),
            beta_schedule=model_cfg.get("ddpm_beta_schedule", "linear"),
            sample_steps=model_cfg.get("ddpm_sample_steps", 1000),
        )
