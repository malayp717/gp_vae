"""Package CLI entry-point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.config.loader import load_config
from src.pipelines import get_pipeline_family
from src.runtime.paths import get_model_type
from src.training.checkpoints import latest_checkpoint_path

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def loss_overrides(args: argparse.Namespace) -> dict | None:
    loss: dict = {}
    if getattr(args, "lpips", None) is not None:
        loss["lpips_enabled"] = args.lpips
    if getattr(args, "adv", None) is not None:
        loss["adv_enabled"] = args.adv
    return loss or None


def build_overrides(args: argparse.Namespace) -> dict | None:
    overrides: dict = {}
    model_override = getattr(args, "model", None)
    if model_override:
        overrides["model"] = {"type": model_override}
    loss_ov = loss_overrides(args)
    if loss_ov:
        overrides["loss"] = loss_ov
    return overrides or None


def resolve_latest_checkpoint(
    config_path: str | Path | None = None,
    model_override: str | None = None,
) -> Path:
    overrides = {"model": {"type": model_override}} if model_override else None
    config = load_config(config_path, overrides=overrides)
    checkpoint_dir = Path(config.get("paths", {}).get("checkpoint_dir", "./checkpoints"))
    return latest_checkpoint_path(checkpoint_dir, get_model_type(config))


def resolve_model_type(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    model_override: str | None = None,
) -> str:
    if model_override:
        return model_override
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_model_type = ckpt.get("config", {}).get("model", {}).get("type")
        if ckpt_model_type:
            return ckpt_model_type
    config = load_config(config_path)
    return get_model_type(config)


def cmd_train(args: argparse.Namespace) -> None:
    model_type = resolve_model_type(args.config, model_override=getattr(args, "model", None))
    family = get_pipeline_family(model_type)
    if family == "diffusion":
        from src.pipelines.diffusion.engine import train as diffusion_train

        diffusion_train(
            config_path=args.config,
            resume_from=getattr(args, "resume", None),
            config_overrides=build_overrides(args),
        )
        return

    from src.pipelines.vae.engine import train as vae_train

    vae_train(
        config_path=args.config,
        resume_from=getattr(args, "resume", None),
        config_overrides=build_overrides(args),
    )


def cmd_validate(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint or resolve_latest_checkpoint(args.config)
    model_type = resolve_model_type(args.config, checkpoint)
    family = get_pipeline_family(model_type)
    if family == "diffusion":
        from src.pipelines.diffusion.evaluation import run_validation as diffusion_run_validation

        diffusion_run_validation(checkpoint_path=checkpoint, config_path=args.config, split=args.split)
        return

    from src.pipelines.vae.evaluation import run_validation as vae_run_validation

    vae_run_validation(checkpoint_path=checkpoint, config_path=args.config, split=args.split)


def cmd_test(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint or resolve_latest_checkpoint(args.config)
    model_type = resolve_model_type(args.config, checkpoint)
    family = get_pipeline_family(model_type)
    if family == "diffusion":
        from src.pipelines.diffusion.generation import (
            generate_interpolations,
            generate_reconstructions,
            generate_samples,
        )
    else:
        from src.pipelines.vae.generation import (
            generate_interpolations,
            generate_reconstructions,
            generate_samples,
        )

    if args.mode in ("samples", "all"):
        generate_samples(checkpoint, args.config, num_samples=args.num_samples)
    if args.mode in ("reconstructions", "all"):
        generate_reconstructions(checkpoint, args.config, num_images=args.num_reconstructions)
    if args.mode in ("interpolations", "all"):
        generate_interpolations(checkpoint, args.config)


def cmd_run_all(args: argparse.Namespace) -> None:
    overrides = build_overrides(args)
    model_type = resolve_model_type(args.config, model_override=getattr(args, "model", None))
    family = get_pipeline_family(model_type)
    if family == "diffusion":
        from src.pipelines.diffusion.engine import train as pipeline_train
        from src.pipelines.diffusion.evaluation import run_validation as pipeline_run_validation
        from src.pipelines.diffusion.generation import (
            generate_interpolations,
            generate_reconstructions,
            generate_samples,
        )
    else:
        from src.pipelines.vae.engine import train as pipeline_train
        from src.pipelines.vae.evaluation import run_validation as pipeline_run_validation
        from src.pipelines.vae.generation import (
            generate_interpolations,
            generate_reconstructions,
            generate_samples,
        )

    if not args.skip_train:
        checkpoint = pipeline_train(
            config_path=args.config,
            resume_from=getattr(args, "resume", None),
            config_overrides=overrides,
        )
    else:
        try:
            checkpoint = resolve_latest_checkpoint(args.config, getattr(args, "model", None))
        except FileNotFoundError as exc:
            logger.error("%s - run without --skip-train first.", exc)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Running validation on test split ...")
    pipeline_run_validation(checkpoint_path=checkpoint, config_path=args.config, split="test")

    logger.info("=" * 60)
    logger.info("Generating outputs ...")
    generate_samples(checkpoint, args.config, num_samples=64)
    generate_reconstructions(checkpoint, args.config, num_images=16)
    generate_interpolations(checkpoint, args.config)

    logger.info("=" * 60)
    logger.info("All done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vae",
        description="Unified CLI for VAE-family and diffusion-family CIFAR-10 models",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train the configured model family (or resume from checkpoint)")
    p_train.add_argument("--config", default=None, help="Path to YAML config")
    p_train.add_argument("--model", choices=["vae", "gp_vae", "sccd", "ddpm"], default=None, help="Override model type from config")
    p_train.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p_train.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=None)
    p_train.add_argument("--adv", action=argparse.BooleanOptionalAction, default=None)

    p_val = sub.add_parser("validate", help="Evaluate a checkpoint on val/test split")
    p_val.add_argument("--checkpoint", default=None)
    p_val.add_argument("--config", default=None)
    p_val.add_argument("--split", choices=["val", "test"], default="test")

    p_test = sub.add_parser("test", help="Generate samples / reconstructions / interpolations")
    p_test.add_argument("--checkpoint", default=None)
    p_test.add_argument("--config", default=None)
    p_test.add_argument("--mode", choices=["samples", "reconstructions", "interpolations", "all"], default="all")
    p_test.add_argument("--num-samples", type=int, default=64)
    p_test.add_argument("--num-reconstructions", type=int, default=16)

    p_all = sub.add_parser("run-all", help="Train -> validate -> generate in sequence")
    p_all.add_argument("--config", default=None)
    p_all.add_argument("--model", choices=["vae", "gp_vae", "sccd", "ddpm"], default=None)
    p_all.add_argument("--resume", default=None)
    p_all.add_argument("--skip-train", action="store_true", help="Use existing checkpoint")
    p_all.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=None)
    p_all.add_argument("--adv", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))
    handlers = {
        "train": cmd_train,
        "validate": cmd_validate,
        "test": cmd_test,
        "run-all": cmd_run_all,
    }
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    handlers[args.command](args)


if __name__ == "__main__":
    main()

