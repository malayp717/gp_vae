# Beta-VAE and GP-VAE

Research-style VAE training code for CIFAR-10 with a package-first layout under `src/vae`.

The codebase supports:

- `vae`: a convolutional beta-VAE
- `gp_vae`: a patch-based generalized posterior VAE with low-rank covariance structure
- optional LPIPS and adversarial training
- periodic validation artifacts and plain-text train/val metric logs

## Layout

```text
vae/
├── config/
│   └── config.yaml
├── dataset/                      # local dataset cache (gitignored)
├── src/
│   └── vae/
│       ├── cli/
│       ├── config/
│       ├── data/
│       ├── evaluation/
│       ├── latent/
│       ├── losses/
│       ├── models/
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

# validate using the latest periodic checkpoint
python main.py validate --split test

# generate samples / reconstructions / interpolations
python main.py test --mode all

# end-to-end: train -> validate -> generate
python main.py run-all

# alternatively, run the package CLI directly
python src/vae/cli/app.py train
```

## Notebook Workflow

Use `train_notebook.ipynb` to launch training from cells instead of the CLI.

- One cell defines the same top-level arguments exposed by the CLI
- A setup cell adds `src/` to `sys.path` and builds config overrides
- The final execution cell calls `vae.training.engine.train(...)`

## Configuration

All runtime settings live in `config/config.yaml`. The loader now validates config structure through a typed schema in `src/vae/config/schema.py`.

The default CIFAR-10 cache path is `dataset/cifar10/`, which is ignored by Git so it will not interfere with `src/vae/data/`.

Key sections:

- `data`: dataset, image size, batch size, workers, split
- `model`: VAE or GP-VAE architecture settings
- `beta_annealing`: KL schedule policy
- `training`: epochs, optimizer-level training knobs, AMP, early stopping
- `loss`: reconstruction, LPIPS, discriminator settings
- `optimizer` / `scheduler`: AdamW + cosine warmup schedule
- `paths`: checkpoint and output directories
- `logging`: save/eval cadence and output artifact settings

## Outputs

For each model type, outputs are written to:

- `checkpoints/`: periodic checkpoints like `vae_epoch_0005.pt`
- `outputs/<model>/logs/train_stats.txt`: persistent training metrics
- `outputs/<model>/logs/val_stats.txt`: persistent validation metrics
- `outputs/<model>/logs/kl_per_dim_<model>_epoch_XXXX.png`: sorted KL-per-dim chart
- `outputs/<model>/reconstructions/`: validation-time reconstructions
- `outputs/<model>/samples/`: sampled images and interpolations

## Notes

- Runtime code lives in `src/vae`. The root `main.py` only bootstraps `src/` onto `sys.path` and forwards into the package CLI.
- `patch_vae` and `patch_transformer_vae` are not supported runtime model types; the supported values are `vae` and `gp_vae`.
- Smoke tests can be run with `python -m unittest discover -s tests`.
