"""Typed, validated configuration loader."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_typed_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load YAML config, merge overrides, and return a typed schema."""
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return ExperimentConfig.from_dict(raw)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML config and return a normalized dict representation."""
    return load_typed_config(config_path=config_path, overrides=overrides).to_dict()

