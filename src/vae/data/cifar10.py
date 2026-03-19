"""CIFAR-10 data loading with train/val/test splits at configurable resolution."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def _build_transforms(image_size: int, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_cifar10_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Create train, validation, and test loaders for CIFAR-10."""
    data_cfg = config["data"]
    image_size = int(data_cfg.get("image_size", 32))
    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    data_dir = Path(data_cfg.get("data_dir", "./dataset/cifar10"))
    download = bool(data_cfg.get("download", True))
    val_split = float(data_cfg.get("val_split", 0.1))

    train_transform = _build_transforms(image_size, augment=True)
    eval_transform = _build_transforms(image_size, augment=False)

    with warnings.catch_warnings():
        # Torchvision's CIFAR-10 pickle loader emits a NumPy 2.4 deprecation warning
        # that does not affect runtime correctness and cannot be fixed from this codebase.
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
        )
        full_train = CIFAR10(root=str(data_dir), train=True, download=download, transform=train_transform)
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val

    generator = torch.Generator().manual_seed(config.get("training", {}).get("seed", 42))
    train_subset, val_indices = random_split(full_train, [n_train, n_val], generator=generator)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
        )
        val_dataset = CIFAR10(root=str(data_dir), train=True, download=False, transform=eval_transform)
    val_subset = Subset(val_dataset, val_indices.indices)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
        )
        test_dataset = CIFAR10(root=str(data_dir), train=False, download=download, transform=eval_transform)

    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

