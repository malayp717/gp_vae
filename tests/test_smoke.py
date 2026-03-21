"""Refactor-safe smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.app import build_overrides
from src.config.loader import load_config, load_typed_config
from src.losses.adversarial import PatchDiscriminator
from src.models.factory import build_model
from src.models.sccd import SCPDTLoss
from src.pipelines import get_pipeline_family
from src.training.beta_schedule import BetaScheduler
from src.training.checkpoints import latest_checkpoint_path, periodic_checkpoint_path


class RefactorSmokeTests(unittest.TestCase):
    def test_load_config_and_schema(self) -> None:
        config = load_config()
        typed = load_typed_config()
        self.assertEqual(config["model"]["type"], typed.model.type)
        self.assertEqual(config["logging"]["save_interval"], typed.logging.save_interval)
        self.assertEqual(config["logging"]["metrics_interval"], typed.logging.metrics_interval)
        self.assertEqual(config["logging"]["artifact_interval"], typed.logging.artifact_interval)
        self.assertAlmostEqual(config["loss"]["free_bits_nats"], typed.loss.free_bits_nats)
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
                "logging": {"metrics_interval": 7, "artifact_interval": 9},
            }
        )
        self.assertEqual(diffusion_typed.model.sample_steps, 2)
        self.assertEqual(diffusion_typed.model.ddpm_sample_steps, 12)
        self.assertEqual(diffusion_typed.model.ddpm_channel_multipliers, [1, 2])
        self.assertEqual(diffusion_typed.training.consistency_k_steps, 3)
        self.assertAlmostEqual(diffusion_typed.training.ema_decay, 0.995)
        self.assertAlmostEqual(diffusion_typed.loss.consistency_weight, 1.25)
        self.assertEqual(diffusion_typed.logging.metrics_interval, 7)
        self.assertEqual(diffusion_typed.logging.artifact_interval, 9)
        sccd_typed = load_typed_config(
            overrides={
                "model": {
                    "type": "sccd",
                    "sample_steps": 3,
                    "sccd_model_dim": 128,
                    "sccd_num_attn_blocks": 2,
                    "sccd_window_size": 2,
                    "sccd_min_log_sigma": -1.0,
                    "sccd_max_log_sigma": 0.5,
                },
                "loss": {
                    "free_bits_nats": 0.05,
                    "sccd_lambda_recon": 1.2,
                    "sccd_lambda_boundary": 0.2,
                    "sccd_lambda_rank": 0.05,
                    "sccd_lambda_sigma": 0.07,
                    "sccd_sigma_reg_target": 0.4,
                },
            }
        )
        self.assertEqual(sccd_typed.model.sccd_model_dim, 128)
        self.assertEqual(sccd_typed.model.sccd_num_attn_blocks, 2)
        self.assertEqual(sccd_typed.model.sample_steps, 3)
        self.assertAlmostEqual(sccd_typed.model.sccd_min_log_sigma, -1.0)
        self.assertAlmostEqual(sccd_typed.model.sccd_max_log_sigma, 0.5)
        self.assertAlmostEqual(sccd_typed.loss.free_bits_nats, 0.05)
        self.assertAlmostEqual(sccd_typed.loss.sccd_lambda_recon, 1.2)
        self.assertAlmostEqual(sccd_typed.loss.sccd_lambda_boundary, 0.2)
        self.assertAlmostEqual(sccd_typed.loss.sccd_lambda_rank, 0.05)
        self.assertAlmostEqual(sccd_typed.loss.sccd_lambda_sigma, 0.07)
        self.assertAlmostEqual(sccd_typed.loss.sccd_sigma_reg_target, 0.4)

    def test_build_model_variants(self) -> None:
        vae_cfg = load_config(overrides={"model": {"type": "vae"}})
        gp_cfg = load_config(overrides={"model": {"type": "gp_vae"}})
        sccd_cfg = load_config(overrides={"model": {"type": "sccd", "sccd_model_dim": 128, "sccd_num_attn_blocks": 2}})
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

    def test_sccd_woodbury_loss_is_finite(self) -> None:
        loss_fn = SCPDTLoss(patch_size=2)
        x0 = torch.randn(2, 3, 4, 4)
        mu = torch.randn(2, 4, 12)
        sigma = torch.rand(2, 4, 12) + 0.1
        V = torch.randn(2, 4, 12, 3)

        losses = loss_fn(x0, mu, sigma, V)

        self.assertIn("total", losses)
        self.assertIn("sigma_mean", losses)
        self.assertIn("sigma_min", losses)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_sccd_sigma_penalty_activates_for_tiny_sigma(self) -> None:
        loss_fn = SCPDTLoss(patch_size=2, lambda_sigma=0.5, sigma_reg_target=0.3)
        x0 = torch.zeros(1, 3, 4, 4)
        mu = torch.zeros(1, 4, 12)
        sigma = torch.full((1, 4, 12), 0.05)
        V = torch.zeros(1, 4, 12, 2)

        losses = loss_fn(x0, mu, sigma, V)

        self.assertGreater(losses["sigma_penalty"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_vae_loss_free_bits_prevents_zero_kl(self) -> None:
        from src.losses import VAELoss

        loss_fn = VAELoss(free_bits_nats=0.1)
        x = torch.zeros(2, 3, 4, 4)
        mu = torch.zeros(2, 8)
        log_var = torch.zeros(2, 8)

        loss, comps = loss_fn(x, x, mu, log_var, beta=1.0)

        self.assertAlmostEqual(comps["kl"].item(), 0.8, places=5)
        self.assertTrue(torch.isfinite(loss))

    def test_patch_discriminator_uses_spectral_norm(self) -> None:
        disc = PatchDiscriminator(in_channels=3, base_channels=16, n_layers=2)
        convs = [module for module in disc.modules() if isinstance(module, torch.nn.Conv2d)]

        self.assertGreater(len(convs), 0)
        for conv in convs:
            self.assertTrue(hasattr(conv, "weight_u"))


if __name__ == "__main__":
    unittest.main()

