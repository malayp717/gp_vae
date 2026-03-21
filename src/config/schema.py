"""Typed configuration schema for experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _list_of_ints(values: list[Any], field_name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"`{field_name}` must be a non-empty list of integers")
    try:
        return [int(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{field_name}` must contain only integers") from exc


@dataclass(slots=True)
class DataConfig:
    dataset: str = "cifar10"
    image_size: int = 32
    in_channels: int = 3
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    data_dir: str = "./dataset/cifar10"
    download: bool = True
    val_split: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        cfg = cls(**data)
        if cfg.dataset != "cifar10":
            raise ValueError(f"Unsupported dataset: {cfg.dataset!r}")
        if cfg.image_size <= 0:
            raise ValueError("`data.image_size` must be positive")
        if not 0.0 < cfg.val_split < 1.0:
            raise ValueError("`data.val_split` must be in (0, 1)")
        return cfg


@dataclass(slots=True)
class ModelConfig:
    type: str = "vae"
    dropout: float = 0.1
    use_batch_norm: bool = True
    latent_dim: int = 256
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 128, 256])
    decoder_channels: list[int] = field(default_factory=lambda: [256, 128, 128, 64, 32])
    patch_div: int = 2
    latent_dim_per_patch: int = 128
    patch_encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    patch_decoder_channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    covariance_rank: int = 16
    num_classes: int = 10
    diffusion_T: int = 1000
    sample_steps: int = 2
    sccd_patch_size: int = 16
    sccd_model_dim: int = 256
    sccd_num_enc_blocks: int = 2
    sccd_num_attn_blocks: int = 4
    sccd_heads: int = 4
    sccd_head_dim: int = 64
    sccd_rank: int = 8
    sccd_window_size: int = 2
    sccd_global_every: int = 4
    sccd_sampler_mode: str = "hybrid"
    sccd_ddim_eta: float = 0.0
    sccd_rank_gamma: float = 1.5
    sccd_k_min_frac: float = 0.125
    sccd_boundary_hidden: int = 64
    sccd_boundary_width: int = 2
    sccd_min_log_sigma: float = -1.5
    sccd_max_log_sigma: float = 1.5
    sccd_log_sigma_temperature: float = 3.0
    ddpm_base_channels: int = 64
    ddpm_channel_multipliers: list[int] = field(default_factory=lambda: [1, 2, 4])
    ddpm_num_res_blocks: int = 2
    ddpm_dropout: float = 0.0
    ddpm_beta_schedule: str = "linear"
    ddpm_sample_steps: int = 1000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        cfg = cls(**data)
        if cfg.type not in {"vae", "gp_vae", "sccd", "ddpm"}:
            raise ValueError(f"Unsupported model type: {cfg.type!r}")
        if cfg.sample_steps <= 0:
            raise ValueError("`model.sample_steps` must be positive")
        if cfg.ddpm_sample_steps <= 0:
            raise ValueError("`model.ddpm_sample_steps` must be positive")
        if cfg.diffusion_T <= 0:
            raise ValueError("`model.diffusion_T` must be positive")
        if cfg.sccd_patch_size <= 0:
            raise ValueError("`model.sccd_patch_size` must be positive")
        if cfg.sccd_model_dim <= 0:
            raise ValueError("`model.sccd_model_dim` must be positive")
        if cfg.sccd_num_enc_blocks <= 0:
            raise ValueError("`model.sccd_num_enc_blocks` must be positive")
        if cfg.sccd_num_attn_blocks <= 0:
            raise ValueError("`model.sccd_num_attn_blocks` must be positive")
        if cfg.sccd_heads <= 0:
            raise ValueError("`model.sccd_heads` must be positive")
        if cfg.sccd_head_dim <= 0:
            raise ValueError("`model.sccd_head_dim` must be positive")
        if cfg.sccd_rank <= 0:
            raise ValueError("`model.sccd_rank` must be positive")
        if cfg.sccd_window_size <= 0:
            raise ValueError("`model.sccd_window_size` must be positive")
        if cfg.sccd_global_every <= 0:
            raise ValueError("`model.sccd_global_every` must be positive")
        if cfg.sccd_sampler_mode not in {"consistency", "hybrid", "ddim"}:
            raise ValueError("`model.sccd_sampler_mode` must be one of {'consistency', 'hybrid', 'ddim'}")
        if not 0.0 <= cfg.sccd_ddim_eta <= 1.0:
            raise ValueError("`model.sccd_ddim_eta` must be in [0, 1]")
        if cfg.sccd_rank_gamma < 0.0:
            raise ValueError("`model.sccd_rank_gamma` must be non-negative")
        if not 0.0 < cfg.sccd_k_min_frac <= 1.0:
            raise ValueError("`model.sccd_k_min_frac` must be in (0, 1]")
        if cfg.sccd_boundary_hidden <= 0:
            raise ValueError("`model.sccd_boundary_hidden` must be positive")
        if cfg.sccd_boundary_width <= 0:
            raise ValueError("`model.sccd_boundary_width` must be positive")
        if cfg.sccd_min_log_sigma >= cfg.sccd_max_log_sigma:
            raise ValueError("`model.sccd_min_log_sigma` must be less than `model.sccd_max_log_sigma`")
        if cfg.sccd_log_sigma_temperature <= 0.0:
            raise ValueError("`model.sccd_log_sigma_temperature` must be positive")
        if cfg.ddpm_num_res_blocks <= 0:
            raise ValueError("`model.ddpm_num_res_blocks` must be positive")
        if cfg.ddpm_base_channels <= 0:
            raise ValueError("`model.ddpm_base_channels` must be positive")
        if cfg.ddpm_beta_schedule not in {"linear", "cosine"}:
            raise ValueError("`model.ddpm_beta_schedule` must be one of {'linear', 'cosine'}")
        cfg.encoder_channels = _list_of_ints(cfg.encoder_channels, "model.encoder_channels")
        cfg.decoder_channels = _list_of_ints(cfg.decoder_channels, "model.decoder_channels")
        cfg.patch_encoder_channels = _list_of_ints(
            cfg.patch_encoder_channels, "model.patch_encoder_channels"
        )
        cfg.patch_decoder_channels = _list_of_ints(
            cfg.patch_decoder_channels, "model.patch_decoder_channels"
        )
        cfg.ddpm_channel_multipliers = _list_of_ints(
            cfg.ddpm_channel_multipliers, "model.ddpm_channel_multipliers"
        )
        return cfg


@dataclass(slots=True)
class BetaAnnealingConfig:
    enabled: bool = True
    schedule: str = "cyclical"
    beta_min: float = 0.0
    beta_max: float = 2.0
    warmup_epochs: int = 60
    cycle_epochs: int = 30
    n_cycles: int = 4
    ratio_increase: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BetaAnnealingConfig":
        cfg = cls(**data)
        if cfg.schedule not in {"linear", "cosine", "cyclical"}:
            raise ValueError(f"Unsupported beta schedule: {cfg.schedule!r}")
        return cfg


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 200
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 1.0
    seed: int = 42
    mixed_precision: bool = True
    compile_model: bool = False
    early_stopping_patience: int = 20
    ema_decay: float = 0.999
    consistency_k_steps: int = 5
    disc_train_interval_epochs: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        cfg = cls(**data)
        if cfg.epochs <= 0:
            raise ValueError("`training.epochs` must be positive")
        if cfg.consistency_k_steps <= 0:
            raise ValueError("`training.consistency_k_steps` must be positive")
        if not 0.0 < cfg.ema_decay <= 1.0:
            raise ValueError("`training.ema_decay` must be in (0, 1]")
        if cfg.disc_train_interval_epochs <= 0:
            raise ValueError("`training.disc_train_interval_epochs` must be positive")
        return cfg


@dataclass(slots=True)
class LossConfig:
    recon_weight: float = 1.0
    lpips_enabled: bool = False
    lpips_weight: float = 0.25
    lpips_net: str = "alex"
    adv_enabled: bool = False
    adv_weight: float = 0.01
    adv_start_epoch: int = 5
    disc_channels: int = 64
    disc_layers: int = 3
    consistency_weight: float = 1.0
    sccd_lambda_recon: float = 1.0
    sccd_lambda_boundary: float = 0.5
    sccd_lambda_rank: float = 0.25
    sccd_lambda_sigma: float = 1.0
    sccd_sigma_reg_target: float = 0.3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LossConfig":
        cfg = cls(**data)
        if cfg.lpips_net not in {"alex", "vgg"}:
            raise ValueError(f"Unsupported LPIPS backbone: {cfg.lpips_net!r}")
        if cfg.consistency_weight < 0.0:
            raise ValueError("`loss.consistency_weight` must be non-negative")
        if cfg.sccd_lambda_recon < 0.0:
            raise ValueError("`loss.sccd_lambda_recon` must be non-negative")
        if cfg.sccd_lambda_boundary < 0.0:
            raise ValueError("`loss.sccd_lambda_boundary` must be non-negative")
        if cfg.sccd_lambda_rank < 0.0:
            raise ValueError("`loss.sccd_lambda_rank` must be non-negative")
        if cfg.sccd_lambda_sigma < 0.0:
            raise ValueError("`loss.sccd_lambda_sigma` must be non-negative")
        if cfg.sccd_sigma_reg_target <= 0.0:
            raise ValueError("`loss.sccd_sigma_reg_target` must be positive")
        return cfg


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "adamw"
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizerConfig":
        cfg = cls(**data)
        if cfg.name.lower() != "adamw":
            raise ValueError(f"Unsupported optimizer: {cfg.name!r}")
        return cfg


@dataclass(slots=True)
class SchedulerConfig:
    name: str = "cosine_warmup"
    warmup_epochs: int = 5
    min_lr: float = 1.0e-6

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchedulerConfig":
        cfg = cls(**data)
        if cfg.name != "cosine_warmup":
            raise ValueError(f"Unsupported scheduler: {cfg.name!r}")
        return cfg


@dataclass(slots=True)
class PathsConfig:
    checkpoint_dir: str = "./checkpoints"
    resume_from: str | None = None
    output_dir: str = "./outputs"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathsConfig":
        return cls(**data)


@dataclass(slots=True)
class LoggingConfig:
    log_interval: int = 50
    save_interval: int = 5
    eval_interval: int = 5
    metrics_interval: int = 25
    artifact_interval: int = 25
    num_fid_samples: int = 1024
    num_samples: int = 64
    num_reconstructions: int = 16

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        cfg = cls(**data)
        if cfg.eval_interval <= 0:
            raise ValueError("`logging.eval_interval` must be positive")
        if cfg.metrics_interval <= 0:
            raise ValueError("`logging.metrics_interval` must be positive")
        if cfg.artifact_interval <= 0:
            raise ValueError("`logging.artifact_interval` must be positive")
        if cfg.save_interval <= 0:
            raise ValueError("`logging.save_interval` must be positive")
        if cfg.num_fid_samples <= 0:
            raise ValueError("`logging.num_fid_samples` must be positive")
        if cfg.num_samples <= 0:
            raise ValueError("`logging.num_samples` must be positive")
        if cfg.num_reconstructions <= 0:
            raise ValueError("`logging.num_reconstructions` must be positive")
        return cfg


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    beta_annealing: BetaAnnealingConfig
    training: TrainingConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    paths: PathsConfig
    logging: LoggingConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        required_sections = ("data", "model", "training", "paths")
        missing = [section for section in required_sections if section not in raw]
        if missing:
            raise ValueError(f"Config is missing required sections: {missing}")

        return cls(
            data=DataConfig.from_dict(raw.get("data", {})),
            model=ModelConfig.from_dict(raw.get("model", {})),
            beta_annealing=BetaAnnealingConfig.from_dict(raw.get("beta_annealing", {})),
            training=TrainingConfig.from_dict(raw.get("training", {})),
            loss=LossConfig.from_dict(raw.get("loss", {})),
            optimizer=OptimizerConfig.from_dict(raw.get("optimizer", {})),
            scheduler=SchedulerConfig.from_dict(raw.get("scheduler", {})),
            paths=PathsConfig.from_dict(raw.get("paths", {})),
            logging=LoggingConfig.from_dict(raw.get("logging", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.paths.checkpoint_dir)

