"""Simple smoke script for instantiating the GP-VAE model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vae.models.gp_vae import GeneralizedPosteriorVAE


def main() -> None:
    model = GeneralizedPosteriorVAE()
    x = torch.randn(2, 3, 224, 224)
    recon, mu, log_var, z = model(x)
    print("input:", tuple(x.shape))
    print("recon:", tuple(recon.shape))
    print("mu:", tuple(mu.shape))
    print("log_var:", tuple(log_var.shape))
    print("z:", tuple(z.shape))


if __name__ == "__main__":
    main()

