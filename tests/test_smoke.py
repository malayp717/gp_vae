"""Refactor-safe smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.app import build_overrides
from src.config.loader import load_config, load_typed_config
from src.models.factory import build_model
from src.pipelines import get_pipeline_family
from src.training.beta_schedule import BetaScheduler
from src.training.checkpoints import latest_checkpoint_path, periodic_checkpoint_path


class RefactorSmokeTests(unittest.TestCase):
    def test_load_config_and_schema(self) -> None:
        config = load_config()
        typed = load_typed_config()
        self.assertEqual(config["model"]["type"], typed.model.type)
        self.assertEqual(config["logging"]["save_interval"], 5)
        diffusion_typed = load_typed_config(
            overrides={
                "model": {
                    "type": "ddpm",
                    "sample_steps": 2,
                    "ddpm_sample_steps": 12,
                    "ddpm_channel_multipliers": [1, 2],
                },
                "training": {"ema_decay": 0.995, "consistency_k_steps": 3},
                "loss": {"consistency_weight": 1.25},
            }
        )
        self.assertEqual(diffusion_typed.model.sample_steps, 2)
        self.assertEqual(diffusion_typed.model.ddpm_sample_steps, 12)
        self.assertEqual(diffusion_typed.model.ddpm_channel_multipliers, [1, 2])
        self.assertEqual(diffusion_typed.training.consistency_k_steps, 3)
        self.assertAlmostEqual(diffusion_typed.training.ema_decay, 0.995)
        self.assertAlmostEqual(diffusion_typed.loss.consistency_weight, 1.25)

    def test_build_model_variants(self) -> None:
        vae_cfg = load_config(overrides={"model": {"type": "vae"}})
        gp_cfg = load_config(overrides={"model": {"type": "gp_vae"}})
        sccd_cfg = load_config(overrides={"model": {"type": "sccd"}})
        ddpm_cfg = load_config(overrides={"model": {"type": "ddpm", "ddpm_channel_multipliers": [1, 2]}})
        self.assertEqual(build_model(vae_cfg).__class__.__name__, "VAE")
        self.assertEqual(build_model(gp_cfg).__class__.__name__, "GeneralizedPosteriorVAE")
        self.assertEqual(build_model(sccd_cfg).__class__.__name__, "SCCDModel")
        self.assertEqual(build_model(ddpm_cfg).__class__.__name__, "DDPMModel")
        self.assertEqual(get_pipeline_family("vae"), "vae")
        self.assertEqual(get_pipeline_family("gp_vae"), "vae")
        self.assertEqual(get_pipeline_family("sccd"), "diffusion")
        self.assertEqual(get_pipeline_family("ddpm"), "diffusion")

    def test_beta_scheduler(self) -> None:
        scheduler = BetaScheduler(load_config())
        self.assertIsInstance(scheduler(0), float)

    def test_cli_overrides(self) -> None:
        class Args:
            model = "ddpm"
            lpips = True
            adv = False

        overrides = build_overrides(Args())
        self.assertEqual(
            overrides,
            {"model": {"type": "ddpm"}, "loss": {"lpips_enabled": True, "adv_enabled": False}},
        )

    def test_checkpoint_layout_helpers(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            epoch5 = periodic_checkpoint_path(checkpoint_dir, "gp_vae", 4)
            epoch10 = periodic_checkpoint_path(checkpoint_dir, "gp_vae", 9)
            epoch5.touch()
            epoch10.touch()

            self.assertEqual(epoch5.parent.name, "gp_vae")
            self.assertEqual(epoch5.name, "gp_vae_epoch_0005.pt")
            self.assertEqual(latest_checkpoint_path(checkpoint_dir, "gp_vae"), epoch10)


if __name__ == "__main__":
    unittest.main()

