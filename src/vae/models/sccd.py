"""Structured-Covariance Consistency Diffusion (SCCD).

Combines a GP-VAE-style encoder/decoder backbone with a consistency-distilled
Diffusion Transformer denoiser. The model file owns only the SCCD model
definition and its runtime primitives; training, evaluation, and CLI routing
live in the diffusion pipeline modules.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.losses.kl import low_rank_kl_per_dim
from vae.models.base import SupportsVAEAPI


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class CosineNoiseSchedule(nn.Module):
    """Cosine beta schedule (Nichol & Dhariwal, 2021).

    Buffers (not parameters – never trained):
        alphas_bar : cumulative product ᾱ_t,  shape (T,)
    """

    def __init__(self, T: int = 1000, s: float = 0.008) -> None:
        super().__init__()
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        alphas_cumprod = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = (1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(0.0, 0.9999)
        self.register_buffer("alphas_bar", torch.cumprod(1.0 - betas, dim=0))
        self.T = T

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        L: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Structured forward process.

        z_t = √ᾱ_t · z0 + √(1−ᾱ_t) · L ε,   ε ~ N(0, I)

        Args:
            z0: Clean latents  (B, N, d).
            t:  Timestep indices  (B,).
            L:  Cholesky factor (B, N, d, d).  If None, uses isotropic noise.

        Returns:
            z_t: Noisy latents  (B, N, d).
        """
        ab = self.alphas_bar[t][:, None, None]          # (B, 1, 1)
        eps = torch.randn_like(z0)                       # (B, N, d)
        if L is not None:
            eps = torch.einsum("bnij,bnj->bni", L, eps) # structured noise
        return ab.sqrt() * z0 + (1.0 - ab).sqrt() * eps


# ---------------------------------------------------------------------------
# Encoder sub-modules  (mirrors gp_vae.py style)
# ---------------------------------------------------------------------------

def _make_patch_cnn(
    in_channels: int,
    channels: Sequence[int],
    use_bn: bool = True,
) -> nn.Sequential:
    """Strided CNN that encodes a single patch to a feature vector."""
    layers: list[nn.Module] = []
    current = in_channels
    for ch in channels:
        layers.append(nn.Conv2d(current, ch, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(ch))
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(2))
        current = ch
    return nn.Sequential(*layers)


class _LatentRefinement(nn.Module):
    """Windowed self-attention refinement over patch latents (from gp_vae.py)."""

    def __init__(
        self,
        latent_dim: int,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            block, layers, enable_nested_tensor=False
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.transformer(z)


# ---------------------------------------------------------------------------
# Decoder sub-modules  (mirrors gp_vae.py style)
# ---------------------------------------------------------------------------

class _PatchTokenDecoder(nn.Module):
    """Decode a single latent token into a patch image (from gp_vae.py)."""

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
        self.fc = nn.Linear(
            token_dim,
            self._init_channels * self._init_size * self._init_size,
        )
        layers: list[nn.Module] = []
        channels_in = self._init_channels
        for channels_out in hidden_channels[1:]:
            layers.append(
                nn.ConvTranspose2d(channels_in, channels_out, 4, stride=2, padding=1)
            )
            layers.append(nn.GELU())
            channels_in = channels_out
        layers.append(nn.Conv2d(channels_in, out_channels, 3, padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(token)
        hidden = hidden.view(
            -1, self._init_channels, self._init_size, self._init_size
        )
        hidden = self.deconv(hidden)
        if hidden.shape[-1] != self.patch_size:
            hidden = F.interpolate(
                hidden,
                size=(self.patch_size, self.patch_size),
                mode="bilinear",
                align_corners=False,
            )
        return hidden


def _build_seam_mask(
    image_size: int, patch_size: int, width: int = 2
) -> torch.Tensor:
    """Binary mask marking patch boundaries (from gp_vae.py)."""
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


# ---------------------------------------------------------------------------
# DiT denoiser sub-modules
# ---------------------------------------------------------------------------

class _TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class _AdaLN(nn.Module):
    """Adaptive LayerNorm: scale and shift from a conditioning vector."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim)  → scale/shift: (B, 1, dim)
        scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class _DiTBlock(nn.Module):
    """Single DiT block: adaLN → MHSA → adaLN → FFN, with residuals."""

    def __init__(
        self,
        dim: int,
        heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = _AdaLN(dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x, cond))
        return x


class _DiTDenoiser(nn.Module):
    """Diffusion Transformer denoiser.

    Maps (z_t, t, c) → predicted clean latent ẑ₀.

    Args:
        num_patches:  Number of spatial patch tokens N.
        latent_dim:   Per-patch latent dimension d.
        depth:        Number of DiT blocks.
        heads:        Attention heads per block.
        num_classes:  Number of class labels (CIFAR-10 → 10).
        dropout:      Dropout inside attention and FFN.
    """

    def __init__(
        self,
        num_patches: int,
        latent_dim: int,
        depth: int = 6,
        heads: int = 4,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cond_dim = latent_dim

        self.input_proj = nn.Linear(latent_dim, latent_dim)
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patches, latent_dim) * 0.02
        )
        self.t_emb = _TimestepEmbedding(latent_dim)
        self.c_emb = nn.Embedding(num_classes + 1, latent_dim)  # +1 for null CFG token

        self.blocks = nn.ModuleList([
            _DiTBlock(latent_dim, heads, cond_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(latent_dim)
        self.final_proj = nn.Linear(latent_dim, latent_dim)

        # Zero-init output projection (standard DiT practice)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t: Noisy latents  (B, N, d).
            t:   Timestep indices  (B,).
            c:   Class labels  (B,).

        Returns:
            Predicted clean latent  (B, N, d).
        """
        x = self.input_proj(z_t) + self.pos_emb          # (B, N, d)
        cond = self.t_emb(t) + self.c_emb(c)             # (B, d)
        for block in self.blocks:
            x = block(x, cond)
        return self.final_proj(self.final_norm(x))        # (B, N, d)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SCCDModel(nn.Module):
    """Structured-Covariance Consistency Diffusion model.

    Implements the SupportsVAEAPI protocol so it can be plugged into the
    existing training, evaluation, and generation infrastructure unchanged.

    ``forward()`` runs the full training graph and returns the standard
    ``(recon, mu, log_var, z)`` 4-tuple that VAELoss expects.  It also
    caches ``_cached_kl`` and ``_cached_kl_per_dim`` exactly like
    GeneralizedPosteriorVAE does.

    The consistency loss lives in ``consistency_loss()`` and is called
    separately by the Phase-2 training loop so the composite VAELoss
    can remain unmodified.

    Args:
        image_size:          Square input image side length.
        patch_div:           Number of patches per side (image_size // patch_div = patch_size).
        in_channels:         Input image channels.
        latent_dim:          Per-patch latent dimension d.
        covariance_rank:     Low-rank factor k in Σ = Diag(σ²) + VVᵀ.
        encoder_channels:    CNN channel widths for the patch encoder.
        decoder_channels:    Transposed-CNN channel widths for the patch decoder.
        transformer_dim:     Hidden dim for the cross-patch transformer encoder/decoder.
        transformer_heads:   Number of attention heads in those transformers.
        transformer_layers:  Depth of those transformers.
        transformer_dropout: Dropout in those transformers.
        dit_depth:           Number of DiT blocks in the denoiser.
        dit_heads:           Attention heads per DiT block.
        dit_dropout:         Dropout inside DiT blocks.
        num_classes:         Number of class labels for the DiT conditioning.
        T:                   Total diffusion timesteps.
        use_batch_norm:      Toggle BatchNorm in the patch CNN.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_div: int = 2,
        in_channels: int = 3,
        latent_dim: int = 64,
        covariance_rank: int = 8,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        transformer_dropout: float = 0.1,
        dit_depth: int = 6,
        dit_heads: int = 4,
        dit_dropout: float = 0.1,
        num_classes: int = 10,
        T: int = 1000,
        sample_steps: int = 1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_div = patch_div
        self.patch_size = image_size // patch_div
        self.num_patches = patch_div ** 2
        self.latent_dim = latent_dim
        self.cov_rank = covariance_rank
        self.sample_steps = int(sample_steps)

        # ── Encoder ────────────────────────────────────────────────────────
        self.patch_cnn = _make_patch_cnn(
            in_channels, encoder_channels, use_bn=use_batch_norm
        )
        feat_dim = encoder_channels[-1]
        self.enc_proj = nn.Linear(feat_dim, transformer_dim)
        self.encoder_pos = nn.Parameter(
            torch.randn(1, self.num_patches, transformer_dim) * 0.02
        )
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
            enc_layer, transformer_layers, enable_nested_tensor=False
        )
        self.fc_mu = nn.Linear(transformer_dim, latent_dim)
        self.fc_log_sigma = nn.Linear(transformer_dim, latent_dim)
        self.fc_V = nn.Linear(transformer_dim, latent_dim * covariance_rank)

        # ── Latent refinement (post-reparameterisation MHSA) ───────────────
        self.global_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.latent_pos = nn.Parameter(
            torch.randn(1, self.num_patches + 1, latent_dim) * 0.02
        )
        latent_heads = min(transformer_heads, latent_dim)
        while latent_heads > 1 and (latent_dim % latent_heads) != 0:
            latent_heads -= 1
        self.latent_refine = _LatentRefinement(
            latent_dim,
            heads=max(1, latent_heads),
            layers=2,
            dropout=transformer_dropout,
        )

        # ── Decoder ────────────────────────────────────────────────────────
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
            dec_layer, transformer_layers, enable_nested_tensor=False
        )
        self.patch_decoder = _PatchTokenDecoder(
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
            _build_seam_mask(image_size, self.patch_size, width=2),
            persistent=False,
        )

        # ── Diffusion schedule & DiT denoiser ──────────────────────────────
        self.schedule = CosineNoiseSchedule(T=T)
        self.denoiser = _DiTDenoiser(
            num_patches=self.num_patches,
            latent_dim=latent_dim,
            depth=dit_depth,
            heads=dit_heads,
            num_classes=num_classes,
            dropout=dit_dropout,
        )
        # EMA copy of the denoiser used as consistency training target
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        for p in self.ema_denoiser.parameters():
            p.requires_grad_(False)

        # ── KL cache (mirrors GeneralizedPosteriorVAE) ─────────────────────
        self._cached_kl: torch.Tensor | None = None
        self._cached_kl_per_dim: torch.Tensor | None = None

    # -----------------------------------------------------------------------
    # Patchify / unpatchify  (identical to gp_vae.py)
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------------------

    def encode_distribution(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, log_sigma, V) for each patch.

        Shapes:
            mu, log_sigma : (B, N, d)
            V             : (B, N, d, k)
        """
        batch = x.size(0)
        patches = self.patchify(x)
        flat = patches.reshape(
            batch * self.num_patches, patches.size(2), self.patch_size, self.patch_size
        )
        feats = self.patch_cnn(flat).mean([2, 3])            # (B*N, feat_dim)
        feats = feats.view(batch, self.num_patches, -1)

        tokens = self.enc_proj(feats) + self.encoder_pos
        ctx = self.transformer_encoder(tokens)               # (B, N, transformer_dim)

        mu = self.fc_mu(ctx)                                 # (B, N, d)
        log_sigma = self.fc_log_sigma(ctx)                   # (B, N, d)
        V = self.fc_V(ctx).view(
            batch, self.num_patches, self.latent_dim, self.cov_rank
        )                                                    # (B, N, d, k)
        return mu, log_sigma, V

    def encode_latent_mean(self, x: torch.Tensor) -> torch.Tensor:
        mu, _, _ = self.encode_distribution(x)
        return mu

    def _cholesky(
        self, sigma: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Cholesky of Σ = Diag(σ²) + VVᵀ.

        Args:
            sigma : (B, N, d)  standard deviations (not squared).
            V     : (B, N, d, k)

        Returns:
            L : (B, N, d, d)  lower-triangular Cholesky factor.
        """
        sigma2_diag = torch.diag_embed(sigma ** 2)           # (B, N, d, d)
        VVT = torch.einsum("bndk,bnek->bnde", V, V)         # (B, N, d, d)
        Sigma = sigma2_diag + VVT
        jitter = 1e-5 * torch.eye(
            self.latent_dim, device=sigma.device, dtype=sigma.dtype
        ).unsqueeze(0).unsqueeze(0)
        return torch.linalg.cholesky(Sigma + jitter)         # (B, N, d, d)

    def reparam(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterised sample using the low-rank + diagonal trick.

        z = μ + σ ⊙ ε₁ + V ε₂,   ε₁ ~ N(0,I_d),  ε₂ ~ N(0,I_k)

        Mirrors the formula in the original gp_vae.py exactly.
        """
        sigma = torch.exp(log_sigma)
        eps1 = torch.randn_like(mu)
        eps2 = torch.randn(
            *mu.shape[:-1], self.cov_rank, device=mu.device, dtype=mu.dtype
        )
        return mu + sigma * eps1 + (V @ eps2.unsqueeze(-1)).squeeze(-1)

    # -----------------------------------------------------------------------
    # Latent refinement (global token + MHSA)
    # -----------------------------------------------------------------------

    def _refine(self, z: torch.Tensor) -> torch.Tensor:
        """Normalise → prepend global token → MHSA → strip global token."""
        batch = z.size(0)
        z = z / (z.std(dim=-1, keepdim=True) + 1e-6)
        global_token = self.global_token.expand(batch, -1, -1)
        z = torch.cat([global_token, z], dim=1) + self.latent_pos
        z = self.latent_refine(z)
        return z[:, 1:]                                      # strip global token

    # -----------------------------------------------------------------------
    # Decoder
    # -----------------------------------------------------------------------

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode refined latent tokens to a full image.

        Args:
            z: (B, N, d) refined latent tokens.

        Returns:
            Reconstructed image  (B, C, H, W)  in [0, 1].
        """
        batch = z.size(0)
        tokens = self.transformer_decoder(self.dec_proj(z))  # (B, N, transformer_dim)
        tokens_flat = tokens.reshape(batch * self.num_patches, -1)
        patches = self.patch_decoder(tokens_flat)            # (B*N, C, p, p)
        patches = patches.view(
            batch, self.num_patches, patches.size(1), self.patch_size, self.patch_size
        )
        image = self.unpatchify(patches)                     # (B, C, H, W)
        delta = self.seam_refiner(image)
        image = image + self.seam_mask.to(dtype=image.dtype, device=image.device) * delta
        return torch.sigmoid(image)

    # -----------------------------------------------------------------------
    # Forward  (VAELoss-compatible 4-tuple)
    # -----------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase-1 / VAE forward pass.

        Encodes → reparameterises → refines → decodes.
        Caches KL for VAELoss.get_kl_override().

        Returns:
            recon   : (B, C, H, W) reconstructed image in [0, 1].
            mu      : (B, N, d) posterior means.
            log_var : (B, N, d) = 2 * log_sigma  (convention matches vae.py).
            z       : (B, N, d) refined latent tokens (global token stripped).
        """
        mu, log_sigma, V = self.encode_distribution(x)
        z = self.reparam(mu, log_sigma, V)
        z = self._refine(z)
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

    # -----------------------------------------------------------------------
    # Consistency diffusion
    # -----------------------------------------------------------------------

    def diffusion_forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        k_steps: int = 5,
    ) -> dict[str, torch.Tensor]:
        """Phase-2 forward pass adding the consistency loss term.

        Runs the GP-VAE encode/decode path (same as forward()) and
        additionally computes the consistency denoising loss.  The
        combined loss dict is consumed by the Phase-2 training loop;
        the recon / KL terms are still passed to VAELoss normally.

        Args:
            x:       Input images  (B, C, H, W).
            c:       Class labels  (B,).
            k_steps: Maximum consistency gap between t and t′.

        Returns:
            Dict with keys ``recon``, ``mu``, ``log_var``, ``z``,
            ``loss_cons``, ``z_hat``.
        """
        mu, log_sigma, V = self.encode_distribution(x)
        sigma = torch.exp(log_sigma)
        L = self._cholesky(sigma, V)                         # (B, N, d, d)

        z_raw = self.reparam(mu, log_sigma, V)
        z_hat = self._refine(z_raw)                          # clean latent
        recon = self.decode(z_hat)

        kl_tokens, kl_per_dim_tokens = low_rank_kl_per_dim(
            mu.reshape(-1, self.latent_dim),
            log_sigma.reshape(-1, self.latent_dim),
            V.reshape(-1, self.latent_dim, self.cov_rank),
        )
        self._cached_kl = kl_tokens * self.num_patches
        self._cached_kl_per_dim = kl_per_dim_tokens * self.num_patches

        # Sample timestep pair (t, t′) with t > t′
        B = x.size(0)
        t = torch.randint(1, self.schedule.T + 1, (B,), device=x.device)
        dt = torch.randint(1, k_steps + 1, (B,), device=x.device)
        t_prime = (t - dt).clamp(min=0)

        z_t = self.schedule.q_sample(z_hat, t, L)
        z_t_prime = self.schedule.q_sample(z_hat, t_prime, L)

        # Online denoiser prediction
        z0_pred = self.denoiser(z_t, t, c)                   # (B, N, d)

        # EMA target (no gradient)
        with torch.no_grad():
            z0_target = self.ema_denoiser(z_t_prime, t_prime, c)

        loss_cons = F.mse_loss(z0_pred, z0_target)

        log_var = 2.0 * log_sigma
        return {
            "recon":     recon,
            "mu":        mu,
            "log_var":   log_var,
            "z":         z_hat,
            "loss_cons": loss_cons,
            "z_hat":     z_hat,
        }

    @torch.no_grad()
    def update_ema(self, decay: float = 0.999) -> None:
        """Exponential moving-average update of the EMA denoiser."""
        for p_ema, p in zip(
            self.ema_denoiser.parameters(), self.denoiser.parameters()
        ):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    def denoiser_params(self) -> Iterator[nn.Parameter]:
        """Yield only the online denoiser's parameters (for separate LR)."""
        yield from self.denoiser.parameters()

    def vae_params(self) -> Iterator[nn.Parameter]:
        """Yield encoder + decoder parameters (excludes denoiser and EMA)."""
        excluded = {
            *self.denoiser.parameters(),
            *self.ema_denoiser.parameters(),
            *self.schedule.parameters(),
        }
        for p in self.parameters():
            if p not in excluded:
                yield p

    # -----------------------------------------------------------------------
    # Sampling  (SupportsVAEAPI)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: torch.device,
        c: torch.Tensor | None = None,
        steps: int | None = None,
        temperature: float = 1.0,
        truncation: float | None = None,
    ) -> torch.Tensor:
        """Generate images via consistency sampling.

        Args:
            n_samples:  Number of images to generate.
            device:     Target device.
            c:          Class label tensor  (n_samples,).  Defaults to zeros.
            steps:      Number of denoising steps (1 = single-step consistency).
            temperature: Scale applied to the initial noise.
            truncation:  If set, clamp initial noise to ±truncation.

        Returns:
            Generated images  (n_samples, C, H, W)  in [0, 1].
        """
        if c is None:
            c = torch.zeros(n_samples, dtype=torch.long, device=device)
        if steps is None:
            steps = self.sample_steps

        z = torch.randn(
            n_samples, self.num_patches, self.latent_dim, device=device
        ) * float(temperature)
        if truncation is not None:
            z = z.clamp(-float(truncation), float(truncation))

        T_buf = torch.full((n_samples,), self.schedule.T, device=device)

        if steps == 1:
            z0 = self.denoiser(z, T_buf, c)
        else:
            ts = torch.linspace(
                self.schedule.T, 0, steps + 1, device=device
            ).long()
            for i in range(steps):
                t_cur = ts[i].expand(n_samples)
                z0 = self.denoiser(z, t_cur, c)
                if i < steps - 1:
                    t_next = ts[i + 1].expand(n_samples)
                    z = self.schedule.q_sample(z0, t_next)   # re-noise (no L → isotropic)

        z0 = self._refine(z0)
        return self.decode(z0)

    # -----------------------------------------------------------------------
    # SupportsVAEAPI helpers
    # -----------------------------------------------------------------------

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alias for encode_distribution (matches gp_vae.py convention)."""
        return self.encode_distribution(x)

    def get_kl_override(self) -> torch.Tensor | None:
        return self._cached_kl

    def get_kl_per_dim(self) -> torch.Tensor | None:
        return self._cached_kl_per_dim

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SCCDModel":
        """Construct from the same config dict format used by build_model()."""
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        return cls(
            image_size=data_cfg.get("image_size", 32),
            patch_div=model_cfg.get("patch_div", 2),
            in_channels=data_cfg.get("in_channels", 3),
            latent_dim=model_cfg.get("latent_dim_per_patch", 64),
            covariance_rank=model_cfg.get("covariance_rank", 8),
            encoder_channels=model_cfg.get("patch_encoder_channels", [32, 64, 128, 256]),
            decoder_channels=model_cfg.get("patch_decoder_channels", [256, 128, 64, 32]),
            transformer_dim=model_cfg.get("transformer_dim", 256),
            transformer_heads=model_cfg.get("transformer_heads", 8),
            transformer_layers=model_cfg.get("transformer_layers", 4),
            transformer_dropout=model_cfg.get("transformer_dropout", 0.1),
            dit_depth=model_cfg.get("dit_depth", 6),
            dit_heads=model_cfg.get("dit_heads", 4),
            dit_dropout=model_cfg.get("dit_dropout", 0.1),
            num_classes=model_cfg.get("num_classes", 10),
            T=model_cfg.get("diffusion_T", 1000),
            sample_steps=model_cfg.get("sample_steps", 1),
            use_batch_norm=model_cfg.get("use_batch_norm", True),
        )