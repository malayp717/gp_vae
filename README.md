# Beta-VAE, GP-VAE, SC-PDT (`sccd`), and DDPM

Research-style generative modeling code for CIFAR-10 with a package-first layout under `src/`.

The codebase supports:

- `vae`: a convolutional beta-VAE
- `gp_vae`: a patch-based generalized posterior VAE with low-rank covariance structure
- `sccd`: a Structured-Covariance Patch Diffusion Transformer (SC-PDT) with per-patch low-rank covariance and spatial attention
- `ddpm`: an unconditional pixel-space DDPM baseline with a timestep-conditioned U-Net denoiser
- optional LPIPS and adversarial training
- periodic validation artifacts and plain-text train/val metric logs

For CIFAR-10, the default runtime keeps the native `32x32x3` image size. Patch-based models use a `2x2` patch grid by default, which gives `4` total patches of size `16x16x3`.

Internally, the repo now uses:

- a VAE pipeline for `vae` and `gp_vae`
- a diffusion pipeline for `sccd` and `ddpm`
- shared runtime, data, checkpoint, and utility modules where behavior is common

## Layout

```text
vae/
├── config/
│   └── config.yaml
├── dataset/                      # local dataset cache (gitignored)
├── src/
│   └── src/
│       ├── cli/
│       ├── config/
│       ├── data/
│       ├── evaluation/
│       ├── latent/
│       ├── losses/
│       ├── models/
│       ├── pipelines/
│       ├── reporting/
│       ├── runtime/
│       ├── training/
│       └── visualization/
├── scripts/
│   └── smoke_gp_vae.py
├── tests/
│   └── test_smoke.py
├── main.py                       # root bootstrap entrypoint
├── train_notebook.ipynb
└── README.md
```

## Quick Start

```bash
# train from the repo root
python main.py train

# train with a model override
python main.py train --model gp_vae

# train the SC-PDT model through the same pipeline
python main.py train --model sccd

# train the vanilla DDPM baseline
python main.py train --model ddpm

# validate using the latest periodic checkpoint
python main.py validate --split test

# generate samples / reconstructions / interpolations
python main.py test --mode all

# end-to-end: train -> validate -> generate
python main.py run-all

# alternatively, run the package CLI directly
python src/cli/app.py train
```

`python main.py train --model sccd` and `python main.py train --model ddpm` use the diffusion pipeline automatically, while `vae` and `gp_vae` use the VAE pipeline. The `--lpips` and `--adv` flags are meaningful for `vae` and `gp_vae`; diffusion-family models ignore them.

## Notebook Workflow

Use `train_notebook.ipynb` to launch training from cells instead of the CLI.

- One cell defines the same top-level arguments exposed by the CLI
- A setup cell adds the repo root to `sys.path` and builds config overrides
- The final execution cell calls `src.training.engine.train(...)`, which dispatches to the VAE or diffusion pipeline based on `model`

## Configuration

All runtime settings live in `config/config.yaml`. The loader now validates config structure through a typed schema in `src/config/schema.py`.

The default CIFAR-10 cache path is `dataset/cifar10/`, which is ignored by Git so it will not interfere with `src/data/`.

Key sections:

- `data`: dataset, image size, batch size, workers, split
- `model`: VAE, GP-VAE, SC-PDT (`sccd`), or DDPM architecture settings
- `beta_annealing`: KL schedule policy for VAE-family models
- `training`: epochs, optimizer-level knobs, AMP, early stopping, and diffusion runtime settings
- `loss`: VAE perceptual/adversarial settings plus SC-PDT's Woodbury/boundary/rank weights
- `optimizer` / `scheduler`: AdamW + cosine warmup schedule
- `paths`: checkpoint and output directories
- `logging`: save/eval cadence and output artifact settings

The encoder/decoder channel lists in `config/config.yaml` are intentionally configurable rather than hard-coded to CIFAR-10. When you move to larger or more complex datasets later, the expected tuning knobs are `data.image_size`, `model.encoder_channels`, `model.decoder_channels`, `model.patch_encoder_channels`, and `model.patch_decoder_channels`.

## Outputs

For each model type, outputs are written to:

- `checkpoints/<model>/`: periodic checkpoints like `checkpoints/vae/vae_epoch_0005.pt`
- `outputs/<model>/logs/train_stats.txt`: persistent training metrics
- `outputs/<model>/logs/val_stats.txt`: persistent validation metrics
- `outputs/<model>/logs/kl_per_dim_<model>_epoch_XXXX.png`: sorted KL-per-dim chart when the pipeline emits KL-per-dim statistics
- `outputs/<model>/reconstructions/`: validation-time reconstructions for VAE-style models and SC-PDT, plus DDPM sample proxies for diffusion-only baselines
- `outputs/<model>/samples/`: sampled images and interpolations

## Notes

- Runtime code lives in `src/`. The root `main.py` forwards into the package CLI.
- `sccd` and `ddpm` are treated as diffusion-family models. `sccd` now uses the SC-PDT patch-space structured covariance objective, while `ddpm` uses pure denoising loss in pixel space.
- `patch_vae` and `patch_transformer_vae` are not supported runtime model types; the supported values are `vae`, `gp_vae`, `sccd`, and `ddpm`.
- Smoke tests can be run with `python -m unittest discover -s tests`.
