"""KL-per-dimension visualization."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def save_kl_per_dim_artifacts(
    kl_per_dim: torch.Tensor,
    model_type: str,
    epoch: int,
    log_dir: Path,
) -> None:
    """Save a sorted KL-per-dimension bar chart PNG."""
    log_dir.mkdir(parents=True, exist_ok=True)

    vec = kl_per_dim.detach().to(dtype=torch.float32, device="cpu")
    values, sorted_idx = torch.sort(vec, descending=True)
    threshold = 0.1
    n_dims = max(int(values.numel()), 1)
    x_positions = list(range(n_dims))
    values_list = values.tolist()
    sorted_idx_list = sorted_idx.tolist()
    colors = ["#2563EB" if value > threshold else "#D1D5DB" for value in values_list]
    active_dims = int(sum(value > threshold for value in values_list))

    fig_width = min(max(10.0, 4.5 + 0.28 * n_dims), 24.0)
    fig, ax = plt.subplots(figsize=(fig_width, 7.5), dpi=220)

    ax.bar(
        x_positions,
        values_list,
        width=0.82,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axhline(
        threshold,
        color="#B45309",
        linestyle="--",
        linewidth=1.6,
        alpha=0.95,
    )

    fig.suptitle(
        f"KL per dimension ({model_type}, epoch {epoch + 1:04d})",
        fontsize=16,
        fontweight="semibold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.942,
        f"Latent dimensions sorted by descending KL. Active dims (> {threshold:.1f} nats): {active_dims}/{n_dims}",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4B5563",
    )
    ax.set_ylabel("KL divergence (nats)", fontsize=12)
    ax.set_xlabel("Sorted latent dimensions", fontsize=12)

    max_val = max(float(values.max().item()) if values.numel() > 0 else 0.0, threshold)
    ax.set_ylim(0.0, max(max_val * 1.12, 0.5))
    ax.set_xlim(-0.7, n_dims - 0.3)

    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    if n_dims <= 16:
        tick_step = 1
    elif n_dims <= 40:
        tick_step = max(2, math.ceil(n_dims / 12))
    else:
        tick_step = 0

    if tick_step > 0:
        tick_positions = list(range(0, n_dims, tick_step))
        tick_labels = [f"z{sorted_idx_list[pos] + 1}" for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=9)
    else:
        ax.set_xticks([])

    legend_handles = [
        Patch(facecolor="#2563EB", edgecolor="none", label="Active dimension"),
        Patch(facecolor="#D1D5DB", edgecolor="none", label="Collapsed dimension"),
        Line2D([0], [0], color="#B45309", linestyle="--", linewidth=1.6, label=f"Threshold ({threshold:.1f} nats)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#E5E7EB",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(
        log_dir / f"kl_per_dim_{model_type}_epoch_{epoch + 1:04d}.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

