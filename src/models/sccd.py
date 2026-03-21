"""Structured-Covariance Patch Diffusion Transformer (SC-PDT) as `sccd`."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineNoiseSchedule(nn.Module):
    """Cosine noise schedule (Nichol & Dhariwal, 2021)."""

    def __init__(self, T: int = 1000, s: float = 0.008) -> None:
        super().__init__()
        self.T = int(T)

        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
        alphas_cumprod = f / f[0]
        betas = (1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(max=0.999)

        self.register_buffer("betas", betas.float(), persistent=False)
        self.register_buffer("alphas", (1.0 - betas).float(), persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod[:-1].float(), persistent=False)
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod[:-1]).float(),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod[:-1]).float(),
            persistent=False,
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward process q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepMLP(nn.Module):
    """Projects sinusoidal timestep embeddings to the model width."""

    def __init__(self, time_dim: int, model_dim: int) -> None:
        super().__init__()
        self.embed = SinusoidalTimestepEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(t))


class PatchMLPBlock(nn.Module):
    """Single MLP block with pre-norm and residual connection."""

    def __init__(self, dim: int, expand: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


def window_partition(x: torch.Tensor, grid_h: int, grid_w: int, win: int) -> torch.Tensor:
    """Partition (B, N, D) tokens arranged in a 2D grid into windows."""
    B, _, D = x.shape
    x = x.view(B, grid_h, grid_w, D)
    x = x.view(B, grid_h // win, win, grid_w // win, win, D)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, win * win, D)


def window_unpartition(x: torch.Tensor, grid_h: int, grid_w: int, win: int, B: int) -> torch.Tensor:
    """Reverse of window_partition."""
    nw_h, nw_w = grid_h // win, grid_w // win
    x = x.view(B, nw_h, nw_w, win, win, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, grid_h * grid_w, -1)


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention using scaled dot-product attention."""

    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        inner = self.heads * self.head_dim
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out(attn)


class SpatialAttentionBlock(nn.Module):
    """Single spatial attention block with windowed/global attention and FFN."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        head_dim: int = 64,
        ff_expand: int = 4,
        is_global: bool = False,
    ) -> None:
        super().__init__()
        self.is_global = is_global
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_expand),
            nn.GELU(),
            nn.Linear(dim * ff_expand, dim),
        )

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int, window_size: int) -> torch.Tensor:
        B = x.size(0)
        residual = x
        h = self.norm1(x)
        if self.is_global:
            h = self.self_attn(h)
        else:
            h = window_partition(h, grid_h, grid_w, window_size)
            h = self.self_attn(h)
            h = window_unpartition(h, grid_h, grid_w, window_size, B)
        x = residual + h
        return x + self.ff(self.norm2(x))


class StructuredCovarianceHead(nn.Module):
    """Predicts per-patch structured covariance parameters."""

    def __init__(
        self,
        model_dim: int,
        patch_dim: int,
        rank: int,
        min_log_sigma: float = -1.5,
        max_log_sigma: float = 1.5,
        log_sigma_temperature: float = 3.0,
    ) -> None:
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.rank = int(rank)
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        self.log_sigma_temperature = float(log_sigma_temperature)
        self.mu_head = nn.Linear(model_dim, patch_dim)
        self.log_sigma_head = nn.Linear(model_dim, patch_dim)
        self.V_head = nn.Linear(model_dim, patch_dim * rank)

    def forward(
        self,
        tokens: torch.Tensor,
        rank_fraction: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = tokens.shape
        d, k = self.patch_dim, self.rank
        mu = self.mu_head(tokens)
        log_sigma_span = self.max_log_sigma - self.min_log_sigma
        temp = max(self.log_sigma_temperature, 1e-6)
        log_sigma_logits = self.log_sigma_head(tokens) / temp
        log_sigma = self.min_log_sigma + log_sigma_span * torch.sigmoid(log_sigma_logits)
        sigma = torch.exp(log_sigma)
        V = self.V_head(tokens).view(B, N, d, k)

        active_k = max(1, min(k, int(round(k * float(rank_fraction)))))
        if active_k < k:
            V = V.clone()
            V[:, :, :, active_k:] = 0.0
        return mu, sigma, V


def structured_sample(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    V: torch.Tensor,
    stochasticity: float = 1.0,
) -> torch.Tensor:
    """Reparameterized sample from N(mu, Diag(sigma) + V V^T)."""
    if stochasticity == 0.0:
        return mu
    B, N, d = mu.shape
    k = V.shape[-1]
    eps1 = torch.randn(B, N, d, device=mu.device, dtype=mu.dtype)
    eps2 = torch.randn(B, N, k, 1, device=mu.device, dtype=mu.dtype)
    diag_component = torch.sqrt(sigma) * eps1
    lr_component = (V @ eps2).squeeze(-1)
    return mu + float(stochasticity) * (diag_component + lr_component)


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, d)."""
    B, C, H, W = images.shape
    p = int(patch_size)
    gh, gw = H // p, W // p
    x = images.view(B, C, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(B, gh * gw, C * p * p)


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    channels: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """(B, N, d) -> (B, C, H, W)."""
    B, _, _ = patches.shape
    p = int(patch_size)
    gh, gw = H // p, W // p
    x = patches.view(B, gh, gw, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, channels, H, W)


class BoundaryRefinementNet(nn.Module):
    """Lightweight ConvNet that smooths patch boundaries."""

    def __init__(self, channels: int = 3, hidden: int = 64, patch_size: int = 64, border_width: int = 8) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.border_width = int(border_width)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )

    def _boundary_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(1, 1, H, W, device=device)
        p, bw = self.patch_size, self.border_width
        for y in range(p, H, p):
            mask[:, :, max(0, y - bw): min(H, y + bw), :] = 1.0
        for x in range(p, W, p):
            mask[:, :, :, max(0, x - bw): min(W, x + bw)] = 1.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        mask = self._boundary_mask(H, W, x.device)
        correction = self.net(x)
        return x + mask * correction


class SCPDTLoss(nn.Module):
    """Woodbury-based structured Gaussian loss for patch diffusion."""

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_boundary: float = 0.1,
        lambda_rank: float = 0.01,
        lambda_sigma: float = 0.05,
        patch_size: int = 64,
        sigma_reg_target: float = 0.3,
    ) -> None:
        super().__init__()
        self.lambda_recon = float(lambda_recon)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_rank = float(lambda_rank)
        self.lambda_sigma = float(lambda_sigma)
        self.patch_size = int(patch_size)
        self.sigma_reg_target = float(sigma_reg_target)

    def _woodbury_loss(
        self,
        x0_patches: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        residual = x0_patches - mu
        D_inv = 1.0 / sigma
        D_inv_V = D_inv.unsqueeze(-1) * V
        VtDiV = torch.einsum("bndk,bndj->bnkj", V, D_inv_V)
        rank = V.shape[-1]
        eye_k = torch.eye(rank, device=mu.device, dtype=mu.dtype).view(1, 1, rank, rank)
        batch_shape = V.shape[:-2]
        M = (eye_k + VtDiV).reshape(-1, rank, rank).contiguous()
        jitter = 1e-6
        M = M + jitter * torch.eye(rank, device=mu.device, dtype=mu.dtype).unsqueeze(0)
        L = torch.linalg.cholesky(M)

        D_inv_r = D_inv * residual
        Vt_D_inv_r = torch.einsum("bndk,bnd->bnk", V, D_inv_r).reshape(-1, rank, 1).contiguous()
        y = torch.linalg.solve_triangular(L, Vt_D_inv_r, upper=False)
        x = torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        M_inv_Vt_D_inv_r = x.reshape(*batch_shape, rank)
        correction = torch.einsum("bndk,bnk->bnd", D_inv_V, M_inv_Vt_D_inv_r)
        sigma_inv_r = D_inv_r - correction

        quad_form = (residual * sigma_inv_r).sum(dim=-1)
        log_det_D = torch.log(sigma).sum(dim=-1)
        log_det_M = (2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1))).sum(dim=-1).reshape(*batch_shape)
        gaussian_const = x0_patches.shape[-1] * math.log(2.0 * math.pi)
        nll = 0.5 * (quad_form + log_det_D + log_det_M + gaussian_const)
        nll = nll / max(float(x0_patches.shape[-1]), 1.0)
        return nll.mean()

    def _boundary_loss(self, image: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        _, _, H, W = image.shape
        loss = torch.zeros((), device=image.device, dtype=image.dtype)
        count = 0
        for y in range(p, H, p):
            loss = loss + (image[:, :, y, :] - image[:, :, y - 1, :]).square().mean()
            count += 1
        for x in range(p, W, p):
            loss = loss + (image[:, :, :, x] - image[:, :, :, x - 1]).square().mean()
            count += 1
        return loss / max(count, 1)

    def forward(
        self,
        x0: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        V: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x0_patches = patchify(x0, self.patch_size)
        recon_loss = self._woodbury_loss(x0_patches, mu, sigma, V)
        x0_recon = unpatchify(mu, self.patch_size, x0.shape[1], x0.shape[2], x0.shape[3])
        boundary_loss = self._boundary_loss(x0_recon)
        rank_penalty = V.square().mean()
        sigma_penalty = F.relu(self.sigma_reg_target - sigma).square().mean()
        total = (
            self.lambda_recon * recon_loss
            + self.lambda_boundary * boundary_loss
            + self.lambda_rank * rank_penalty
            + self.lambda_sigma * sigma_penalty
        )
        return {
            "total": total,
            "recon": recon_loss.detach(),
            "boundary": boundary_loss.detach(),
            "rank_penalty": rank_penalty.detach(),
            "sigma_penalty": sigma_penalty.detach(),
            "sigma_mean": sigma.mean().detach(),
            "sigma_min": sigma.amin().detach(),
        }


class SCCDModel(nn.Module):
    """SC-PDT model, exposed through the existing `sccd` model type."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 16,
        channels: int = 3,
        model_dim: int = 256,
        num_enc_blocks: int = 2,
        num_attn_blocks: int = 4,
        heads: int = 4,
        head_dim: int = 64,
        rank: int = 8,
        window_size: int = 2,
        global_every: int = 4,
        diffusion_T: int = 1000,
        sample_steps: int = 2,
        sampler_mode: str = "hybrid",
        ddim_eta: float = 0.0,
        rank_gamma: float = 1.5,
        k_min_frac: float = 0.125,
        boundary_hidden: int = 64,
        boundary_width: int = 2,
        lambda_recon: float = 1.0,
        lambda_boundary: float = 0.1,
        lambda_rank: float = 0.01,
        lambda_sigma: float = 0.05,
        sigma_reg_target: float = 0.3,
        min_log_sigma: float = -1.5,
        max_log_sigma: float = 1.5,
        log_sigma_temperature: float = 3.0,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        self.rank = int(rank)
        self.window_size = int(window_size)
        self.grid_h = self.image_size // self.patch_size
        self.grid_w = self.image_size // self.patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = self.patch_size * self.patch_size * self.channels
        self.sample_steps = int(sample_steps)
        self.sampler_mode = sampler_mode
        self.ddim_eta = float(ddim_eta)
        self.rank_gamma = float(rank_gamma)
        self.k_min_frac = float(k_min_frac)

        if self.image_size % self.patch_size != 0:
            raise ValueError("`image_size` must be divisible by `patch_size` for SC-PDT")
        if self.grid_h % self.window_size != 0 or self.grid_w % self.window_size != 0:
            raise ValueError("SC-PDT window size must divide the patch grid dimensions")

        self.patch_embed = nn.Linear(self.patch_dim, model_dim)
        self.time_embed = TimestepMLP(256, model_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, model_dim) * 0.02)
        self.patch_encoder = nn.Sequential(*[PatchMLPBlock(model_dim) for _ in range(num_enc_blocks)])
        self.attn_blocks = nn.ModuleList(
            [
                SpatialAttentionBlock(
                    dim=model_dim,
                    heads=heads,
                    head_dim=head_dim,
                    is_global=((idx + 1) % max(global_every, 1) == 0),
                )
                for idx in range(num_attn_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.cov_head = StructuredCovarianceHead(
            model_dim,
            self.patch_dim,
            self.rank,
            min_log_sigma=min_log_sigma,
            max_log_sigma=max_log_sigma,
            log_sigma_temperature=log_sigma_temperature,
        )
        self.boundary_refine = BoundaryRefinementNet(
            channels=self.channels,
            hidden=boundary_hidden,
            patch_size=self.patch_size,
            border_width=boundary_width,
        )
        self.schedule = CosineNoiseSchedule(T=diffusion_T)
        self.loss_fn = SCPDTLoss(
            lambda_recon=lambda_recon,
            lambda_boundary=lambda_boundary,
            lambda_rank=lambda_rank,
            lambda_sigma=lambda_sigma,
            patch_size=self.patch_size,
            sigma_reg_target=sigma_reg_target,
        )

    def _rank_fraction(self, t: torch.Tensor) -> float:
        # Use the average timestep across the batch. Using `t.max()` makes the
        # schedule effectively constant for typical batch sizes because the max
        # is almost always near T-1.
        t_value = float(t.float().mean().item()) if t.numel() > 0 else 0.0
        denom = max(float(self.schedule.T - 1), 1.0)
        frac = (t_value / denom) ** self.rank_gamma
        return self.k_min_frac + (1.0 - self.k_min_frac) * frac

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        rank_fraction: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del cond
        if rank_fraction is None:
            rank_fraction = self._rank_fraction(t)
        patches = patchify(x_t, self.patch_size)
        h = self.patch_embed(patches) + self.pos_embed
        h = h + self.time_embed(t).unsqueeze(1)
        h = self.patch_encoder(h)
        for block in self.attn_blocks:
            h = block(h, self.grid_h, self.grid_w, self.window_size)
        h = self.final_norm(h)
        return self.cov_head(h, rank_fraction=rank_fraction)

    @torch.no_grad()
    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma, V = self(x_t, t, rank_fraction=self._rank_fraction(t))
        x0_pred = unpatchify(mu, self.patch_size, self.channels, self.image_size, self.image_size)
        return x0_pred, mu, sigma, V

    def training_loss(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del labels
        B = x0.shape[0]
        t = torch.randint(0, self.schedule.T, (B,), device=x0.device)
        x_t = self.schedule.q_sample(x0, t)
        mu, sigma, V = self(x_t, t, rank_fraction=self._rank_fraction(t))
        losses = self.loss_fn(x0, mu, sigma, V)
        losses["x0_pred"] = unpatchify(mu, self.patch_size, self.channels, x0.shape[2], x0.shape[3])
        return losses

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        x0_pred, _, _, _ = self.predict_x0(x, t)
        return self.boundary_refine(x0_pred).clamp(0.0, 1.0)

    def _sample_ddim(
        self,
        initial: torch.Tensor,
        num_steps: int,
        eta: float,
    ) -> torch.Tensor:
        x = initial
        step_indices = torch.linspace(self.schedule.T - 1, 0, num_steps + 1, device=x.device).long().tolist()
        for idx in range(num_steps):
            t_now = step_indices[idx]
            t_next = step_indices[idx + 1]
            t_tensor = torch.full((x.shape[0],), t_now, device=x.device, dtype=torch.long)
            x0_pred, mu_patches, sigma_patches, V_patches = self.predict_x0(x, t_tensor)

            if t_next == 0:
                x = self.boundary_refine(x0_pred)
                continue

            alpha_now = self.schedule.alphas_cumprod[t_now]
            alpha_next = self.schedule.alphas_cumprod[t_next]
            sqrt_alpha_now = self.schedule.sqrt_alphas_cumprod[t_now]
            sqrt_one_minus_now = self.schedule.sqrt_one_minus_alphas_cumprod[t_now]
            eps_pred = (x - sqrt_alpha_now * x0_pred) / (sqrt_one_minus_now + 1e-8)

            sigma_ddim = (
                eta
                * torch.sqrt((1.0 - alpha_next) / (1.0 - alpha_now + 1e-8))
                * torch.sqrt(torch.clamp(1.0 - alpha_now / (alpha_next + 1e-8), min=0.0))
            )
            dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma_ddim**2, min=0.0))
            x = torch.sqrt(alpha_next) * x0_pred + dir_coeff * eps_pred

            if eta > 0:
                noise_patches = structured_sample(
                    mu=torch.zeros_like(mu_patches),
                    sigma=sigma_patches,
                    V=V_patches,
                    stochasticity=1.0,
                )
                structured_noise = unpatchify(
                    noise_patches,
                    self.patch_size,
                    self.channels,
                    self.image_size,
                    self.image_size,
                )
                x = x + sigma_ddim * structured_noise

        return self.boundary_refine(x).clamp(0.0, 1.0)

    def _sample_consistency(self, initial: torch.Tensor) -> torch.Tensor:
        t = torch.full((initial.shape[0],), self.schedule.T - 1, device=initial.device, dtype=torch.long)
        x0_pred, mu, sigma, V = self.predict_x0(initial, t)
        noise = structured_sample(mu, sigma, V, stochasticity=0.1)
        x0_corrected = unpatchify(
            noise,
            self.patch_size,
            self.channels,
            self.image_size,
            self.image_size,
        )
        return self.boundary_refine(0.5 * x0_pred + 0.5 * x0_corrected).clamp(0.0, 1.0)

    def _sample_hybrid(self, initial: torch.Tensor, num_steps: int) -> torch.Tensor:
        if num_steps <= 1:
            return self._sample_consistency(initial)

        x = initial
        t_start = self.schedule.T - 1
        t_jump = max(1, t_start // num_steps)
        t_tensor = torch.full((x.shape[0],), t_start, device=x.device, dtype=torch.long)
        x0_pred, _, _, _ = self.predict_x0(x, t_tensor)
        alpha_jump = self.schedule.alphas_cumprod[t_jump]
        x = torch.sqrt(alpha_jump) * x0_pred + torch.sqrt(1.0 - alpha_jump) * torch.randn_like(x0_pred)
        remaining_steps = max(num_steps - 1, 1)
        step_indices = torch.linspace(t_jump, 0, remaining_steps + 1, device=x.device).long().tolist()
        for idx in range(remaining_steps):
            t_now = step_indices[idx]
            t_next = step_indices[idx + 1]
            t_now_tensor = torch.full((x.shape[0],), t_now, device=x.device, dtype=torch.long)
            x0_pred, _, _, _ = self.predict_x0(x, t_now_tensor)
            if t_next == 0:
                x = x0_pred
            else:
                x = self.schedule.q_sample(
                    x0_pred,
                    torch.full((x.shape[0],), t_next, device=x.device, dtype=torch.long),
                )
        return self.boundary_refine(x).clamp(0.0, 1.0)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: torch.device,
        c: torch.Tensor | None = None,
        steps: int | None = None,
        temperature: float = 1.0,
        truncation: float | None = None,
        mode: str | None = None,
    ) -> torch.Tensor:
        del c
        sample_steps = self.sample_steps if steps is None else int(steps)
        sample_steps = max(1, sample_steps)
        mode = self.sampler_mode if mode is None else mode
        x = torch.randn(n_samples, self.channels, self.image_size, self.image_size, device=device)
        x = x * float(temperature)
        if truncation is not None:
            x = x.clamp(-float(truncation), float(truncation))

        if mode == "consistency" or sample_steps == 1:
            return self._sample_consistency(x)
        if mode == "hybrid":
            return self._sample_hybrid(x, min(sample_steps, 3))
        return self._sample_ddim(x, sample_steps, self.ddim_eta)

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
            noise_a = torch.randn(1, self.channels, self.image_size, self.image_size, device=device)
            noise_b = torch.randn(1, self.channels, self.image_size, self.image_size, device=device)
            alphas = torch.linspace(0.0, 1.0, n_steps, device=device)
            interp = torch.cat([(1.0 - alpha) * noise_a + alpha * noise_b for alpha in alphas], dim=0)
            rows.append(self._sample_hybrid(interp, sample_steps or self.sample_steps))
        return torch.cat(rows, dim=0)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SCCDModel":
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        loss_cfg = config.get("loss", {})
        image_size = int(data_cfg.get("image_size", 32))
        patch_div = int(model_cfg.get("patch_div", 2))
        patch_size = int(model_cfg.get("sccd_patch_size", image_size // patch_div))
        return cls(
            image_size=image_size,
            patch_size=patch_size,
            channels=int(data_cfg.get("in_channels", 3)),
            model_dim=int(model_cfg.get("sccd_model_dim", 256)),
            num_enc_blocks=int(model_cfg.get("sccd_num_enc_blocks", 2)),
            num_attn_blocks=int(model_cfg.get("sccd_num_attn_blocks", 4)),
            heads=int(model_cfg.get("sccd_heads", 4)),
            head_dim=int(model_cfg.get("sccd_head_dim", 64)),
            rank=int(model_cfg.get("sccd_rank", 8)),
            window_size=int(model_cfg.get("sccd_window_size", 2)),
            global_every=int(model_cfg.get("sccd_global_every", 4)),
            diffusion_T=int(model_cfg.get("diffusion_T", 1000)),
            sample_steps=int(model_cfg.get("sample_steps", 2)),
            sampler_mode=str(model_cfg.get("sccd_sampler_mode", "hybrid")),
            ddim_eta=float(model_cfg.get("sccd_ddim_eta", 0.0)),
            rank_gamma=float(model_cfg.get("sccd_rank_gamma", 1.5)),
            k_min_frac=float(model_cfg.get("sccd_k_min_frac", 0.125)),
            boundary_hidden=int(model_cfg.get("sccd_boundary_hidden", 64)),
            boundary_width=int(model_cfg.get("sccd_boundary_width", 2)),
            lambda_recon=float(loss_cfg.get("sccd_lambda_recon", 1.0)),
            lambda_boundary=float(loss_cfg.get("sccd_lambda_boundary", 0.1)),
            lambda_rank=float(loss_cfg.get("sccd_lambda_rank", 0.01)),
            lambda_sigma=float(loss_cfg.get("sccd_lambda_sigma", 0.05)),
            sigma_reg_target=float(loss_cfg.get("sccd_sigma_reg_target", 0.3)),
            min_log_sigma=float(model_cfg.get("sccd_min_log_sigma", -1.5)),
            max_log_sigma=float(model_cfg.get("sccd_max_log_sigma", 1.5)),
            log_sigma_temperature=float(model_cfg.get("sccd_log_sigma_temperature", 3.0)),
        )
