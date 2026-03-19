"""KL divergence helpers."""

from __future__ import annotations

import torch

_LOG_SIGMA_CLAMP = 10.0
_EPS = 1e-6


def diagonal_kl_per_dim(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Return mean KL contribution for each latent dimension."""
    mu_flat = mu.reshape(-1, mu.shape[-1])
    log_var_flat = log_var.reshape(-1, log_var.shape[-1])
    return 0.5 * (log_var_flat.exp() + mu_flat.pow(2) - 1.0 - log_var_flat).mean(dim=0)


def low_rank_kl(mu: torch.Tensor, log_sigma: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """KL(q||p) where q has diagonal+low-rank covariance and p is N(0, I)."""
    _, dim = mu.shape
    rank = V.shape[2]

    log_sigma = log_sigma.clamp(-_LOG_SIGMA_CLAMP, _LOG_SIGMA_CLAMP)
    sigma_sq = torch.exp(2.0 * log_sigma)

    tr_sigma = sigma_sq.sum(dim=1) + (V * V).sum(dim=(1, 2))
    mu_sq = mu.pow(2).sum(dim=1)
    log_det_diag = (2.0 * log_sigma).sum(dim=1)

    inv_sigma_sq = 1.0 / (sigma_sq + _EPS)
    V_scaled = V * inv_sigma_sq.unsqueeze(2)

    eye = torch.eye(rank, device=mu.device, dtype=mu.dtype).unsqueeze(0)
    matrix = eye + torch.bmm(V.transpose(1, 2), V_scaled)
    matrix = matrix + _EPS * eye

    log_det_matrix = torch.linalg.slogdet(matrix)[1]
    log_det_sigma = log_det_diag + log_det_matrix
    kl = 0.5 * (tr_sigma - dim + mu_sq - log_det_sigma)
    return kl.mean()


def low_rank_kl_per_dim(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    V: torch.Tensor,
    eps: float = _EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return total and per-dimension KL for a diagonal+low-rank posterior."""
    _, dim = mu.shape
    rank = V.shape[2]

    log_sigma = log_sigma.clamp(-_LOG_SIGMA_CLAMP, _LOG_SIGMA_CLAMP)
    sigma_sq = torch.exp(2.0 * log_sigma)
    var_diag = sigma_sq + (V * V).sum(dim=2)
    base = 0.5 * (var_diag + mu.pow(2) - 1.0 - 2.0 * log_sigma)

    inv_sigma_sq = 1.0 / (sigma_sq + eps)
    V_scaled = V * inv_sigma_sq.unsqueeze(2)

    eye = torch.eye(rank, device=mu.device, dtype=mu.dtype).unsqueeze(0)
    matrix = eye + torch.bmm(V.transpose(1, 2), V_scaled)
    matrix = matrix + eps * eye
    log_det_matrix = torch.linalg.slogdet(matrix)[1]

    corr = (-0.5 * log_det_matrix).unsqueeze(1)
    per_dim = base + corr / float(dim)
    kl_mean = per_dim.sum(dim=1).mean()
    per_dim_mean = per_dim.mean(dim=0)
    return kl_mean, per_dim_mean

