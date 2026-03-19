"""Compatibility re-export for VAE-family training loops."""

from vae.pipelines.vae.loops import train_one_epoch, validate_one_epoch

__all__ = ["train_one_epoch", "validate_one_epoch"]

