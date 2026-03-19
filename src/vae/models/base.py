"""Shared model interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SupportsVAEAPI(Protocol):
    """Common interface expected by training, evaluation, and generation code."""

    def encode_latent_mean(self, x: torch.Tensor) -> torch.Tensor: ...
    def get_kl_override(self) -> torch.Tensor | None: ...
    def get_kl_per_dim(self) -> torch.Tensor | None: ...
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor: ...
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...


def get_kl_override(model: torch.nn.Module) -> torch.Tensor | None:
    if isinstance(model, SupportsVAEAPI):
        return model.get_kl_override()
    return getattr(model, "_cached_kl", None)


def get_kl_per_dim(model: torch.nn.Module) -> torch.Tensor | None:
    if isinstance(model, SupportsVAEAPI):
        return model.get_kl_per_dim()
    return getattr(model, "_cached_kl_per_dim", None)


def encode_latent_mean(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, SupportsVAEAPI):
        return model.encode_latent_mean(x)
    mu, _ = model.encode(x)  # type: ignore[attr-defined]
    return mu

