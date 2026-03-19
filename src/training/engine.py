"""Top-level training dispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.loader import load_config
from src.pipelines import get_pipeline_family


def train(
    config_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Route training to the appropriate pipeline for the configured model type."""
    config = load_config(config_path, overrides=config_overrides)
    model_type = config.get("model", {}).get("type", "vae")
    family = get_pipeline_family(model_type)

    if family == "diffusion":
        from src.pipelines.diffusion.engine import train as diffusion_train

        return diffusion_train(
            config_path=config_path,
            resume_from=resume_from,
            config_overrides=config_overrides,
        )

    from src.pipelines.vae.engine import train as vae_train

    return vae_train(
        config_path=config_path,
        resume_from=resume_from,
        config_overrides=config_overrides,
    )

