"""Refactor-safe smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vae.cli.app import build_overrides
from vae.config.loader import load_config, load_typed_config
from vae.models.factory import build_model
from vae.training.beta_schedule import BetaScheduler


class RefactorSmokeTests(unittest.TestCase):
    def test_load_config_and_schema(self) -> None:
        config = load_config()
        typed = load_typed_config()
        self.assertEqual(config["model"]["type"], typed.model.type)
        self.assertEqual(config["logging"]["save_interval"], 5)

    def test_build_model_variants(self) -> None:
        vae_cfg = load_config(overrides={"model": {"type": "vae"}})
        gp_cfg = load_config(overrides={"model": {"type": "gp_vae"}})
        self.assertEqual(build_model(vae_cfg).__class__.__name__, "VAE")
        self.assertEqual(build_model(gp_cfg).__class__.__name__, "GeneralizedPosteriorVAE")

    def test_beta_scheduler(self) -> None:
        scheduler = BetaScheduler(load_config())
        self.assertIsInstance(scheduler(0), float)

    def test_cli_overrides(self) -> None:
        class Args:
            model = "gp_vae"
            lpips = True
            adv = False

        overrides = build_overrides(Args())
        self.assertEqual(
            overrides,
            {"model": {"type": "gp_vae"}, "loss": {"lpips_enabled": True, "adv_enabled": False}},
        )


if __name__ == "__main__":
    unittest.main()

