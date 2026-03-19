"""Pipeline family helpers."""

from __future__ import annotations

VAE_MODEL_TYPES = frozenset({"vae", "gp_vae"})
DIFFUSION_MODEL_TYPES = frozenset({"sccd", "ddpm"})


def get_pipeline_family(model_type: str) -> str:
    """Return the internal pipeline family for a model type."""
    if model_type in VAE_MODEL_TYPES:
        return "vae"
    if model_type in DIFFUSION_MODEL_TYPES:
        return "diffusion"
    raise ValueError(f"Unsupported model type for pipeline routing: {model_type!r}")


def is_diffusion_model(model_type: str) -> bool:
    return get_pipeline_family(model_type) == "diffusion"


__all__ = [
    "DIFFUSION_MODEL_TYPES",
    "VAE_MODEL_TYPES",
    "get_pipeline_family",
    "is_diffusion_model",
]
