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
    image_size: int = 128
    in_channels: int = 3
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    data_dir: str = "./data/cifar10"
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
    patch_div: int = 4
    latent_dim_per_patch: int = 128
    patch_encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    patch_decoder_channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    covariance_rank: int = 16

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        cfg = cls(**data)
        if cfg.type not in {"vae", "gp_vae"}:
            raise ValueError(f"Unsupported model type: {cfg.type!r}")
        cfg.encoder_channels = _list_of_ints(cfg.encoder_channels, "model.encoder_channels")
        cfg.decoder_channels = _list_of_ints(cfg.decoder_channels, "model.decoder_channels")
        cfg.patch_encoder_channels = _list_of_ints(
            cfg.patch_encoder_channels, "model.patch_encoder_channels"
        )
        cfg.patch_decoder_channels = _list_of_ints(
            cfg.patch_decoder_channels, "model.patch_decoder_channels"
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        cfg = cls(**data)
        if cfg.epochs <= 0:
            raise ValueError("`training.epochs` must be positive")
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LossConfig":
        cfg = cls(**data)
        if cfg.lpips_net not in {"alex", "vgg"}:
            raise ValueError(f"Unsupported LPIPS backbone: {cfg.lpips_net!r}")
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
    num_fid_samples: int = 1024
    num_samples: int = 64
    num_reconstructions: int = 16

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        cfg = cls(**data)
        if cfg.save_interval <= 0:
            raise ValueError("`logging.save_interval` must be positive")
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

