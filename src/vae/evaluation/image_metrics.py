"""Image quality metrics."""

from __future__ import annotations

import math
from typing import Any

import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm.auto import tqdm


@torch.no_grad()
def compute_image_metrics(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_fid_samples: int = 1024,
    fid_feature_dim: int = 2048,
) -> dict[str, float]:
    """Compute FID, SSIM, and PSNR for a trained VAE."""
    model.eval()
    fid_metric = FrechetInceptionDistance(feature=fid_feature_dim, normalize=True).to("cpu")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to("cpu")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to("cpu")

    collected_real = 0
    pbar_real = tqdm(loader, desc="Metrics (real+recon)", leave=False)
    for images, _ in pbar_real:
        if collected_real >= num_fid_samples:
            break
        images = images.to(device, non_blocking=True)
        recon, _, _, _ = model(images)
        images_cpu = images.cpu()
        recon_cpu = recon.cpu().clamp(0, 1)
        ssim_metric.update(recon_cpu, images_cpu)
        psnr_metric.update(recon_cpu, images_cpu)
        fid_metric.update(images_cpu, real=True)
        collected_real += images.size(0)
        pbar_real.set_postfix(collected=collected_real)
    pbar_real.close()

    n_generated = 0
    batch_size = loader.batch_size or 64
    pbar_gen = tqdm(range(math.ceil(num_fid_samples / batch_size)), desc="Metrics (samples)", leave=False)
    for _ in pbar_gen:
        n = min(batch_size, num_fid_samples - n_generated)
        samples = model.sample(n, device).cpu().clamp(0, 1)
        fid_metric.update(samples, real=False)
        n_generated += n
    pbar_gen.close()
    return {
        "fid": fid_metric.compute().item(),
        "ssim": ssim_metric.compute().item(),
        "psnr": psnr_metric.compute().item(),
    }

