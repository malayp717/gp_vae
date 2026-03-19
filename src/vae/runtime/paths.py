"""Project path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def get_model_type(config: dict[str, Any]) -> str:
    """Return configured model type (default: 'vae')."""
    return config.get("model", {}).get("type", "vae")


def get_run_output_dir(
    config: dict[str, Any],
    base_output_dir: str | Path | None = None,
) -> Path:
    """Return the per-model output root: ``<output_dir>/<model_type>``."""
    paths_cfg = config.get("paths", {})
    base = Path(base_output_dir or paths_cfg.get("output_dir", "./outputs"))
    return base / get_model_type(config)

