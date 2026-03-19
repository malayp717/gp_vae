# GP-VAE (Generalized Posterior VAE) — Technical Notes

This document describes the current GP-VAE implementation in
`src/models/gp_vae.py` and the structured-KL helpers in
`src/losses/kl.py`.

The current architecture is a patch-latent VAE with:

- a CNN patch encoder
- transformer context over patch tokens
- a diagonal + low-rank posterior covariance
- latent refinement with a learnable global token
- per-patch decoding followed by seam-aware residual refinement

## High-level design

For an image `x` with shape `(C, H, W)`, the model:

1. splits the image into a regular grid of patches
2. encodes each patch into a token
3. predicts posterior parameters per patch token
4. samples a latent token per patch using a structured posterior
5. refines the latent sequence with a small transformer
6. decodes patch tokens back into image patches
7. stitches the patches back together
8. applies a seam-aware refinement layer

Key derived quantities:

- `patch_div = G`
- `patch_size = image_size // patch_div = P`
- `num_patches = G^2 = N`
- per-patch latent width `latent_dim = d`
- low-rank covariance width `covariance_rank = k`

So the latent representation is patch-token based rather than one global latent
vector for the whole image.

## Source locations

- Model class: `src/models/gp_vae.py`
- Structured KL math: `src/losses/kl.py`

Main class:

- `GeneralizedPosteriorVAE`

Main helper components:

- `LatentRefinement`
- `PatchTokenDecoder`
- `make_cnn()`
- `_build_seam_mask()`

Main KL helpers:

- `low_rank_kl()`
- `low_rank_kl_per_dim()`

## Encoder pipeline

The encoder path in `GeneralizedPosteriorVAE` is:

1. `patchify(x)` splits the image into non-overlapping patches
2. each patch goes through `patch_cnn`
3. spatial features are averaged
4. `enc_proj` maps them into `transformer_dim`
5. learnable `encoder_pos` embeddings are added
6. `transformer_encoder` contextualizes the patch-token sequence
7. three posterior heads predict:
   - `mu`
   - `log_sigma`
   - `V`

### Shapes

Let batch size be `B`.

- input patches: `(B, N, C, P, P)`
- encoder tokens: `(B, N, transformer_dim)`
- `mu`: `(B, N, d)`
- `log_sigma`: `(B, N, d)`
- `V`: `(B, N, d, k)`

## Structured posterior

For each patch token, the posterior is:

\[
q(z_i \mid x) = \mathcal{N}(\mu_i, \Sigma_i)
\]

with covariance:

\[
\Sigma_i = \operatorname{diag}(\sigma_i^2) + V_i V_i^\top
\]

where:

- \(\sigma_i = \exp(\log\sigma_i)\)
- \(V_i \in \mathbb{R}^{d \times k}\)

This is more expressive than a fully factorized posterior because it can model
correlated uncertainty directions without paying the cost of a full dense
covariance.

## Reparameterization

Sampling uses:

\[
z_i = \mu_i + \sigma_i \odot \epsilon_{1,i} + V_i \epsilon_{2,i}
\]

where:

- \(\epsilon_{1,i} \sim \mathcal{N}(0, I_d)\)
- \(\epsilon_{2,i} \sim \mathcal{N}(0, I_k)\)

In code, this is implemented by `GeneralizedPosteriorVAE.reparam(...)`.

The low-rank term `(V @ eps2.unsqueeze(-1)).squeeze(-1)` is what introduces
correlated latent noise.

## KL divergence

The prior is standard normal:

\[
p(z) = \mathcal{N}(0, I)
\]

For the structured posterior, the token-wise KL is:

\[
\mathrm{KL}\big(\mathcal{N}(\mu,\Sigma)\,\|\,\mathcal{N}(0,I)\big)
= \tfrac12\left(\operatorname{tr}(\Sigma) + \mu^\top\mu - d - \log\det(\Sigma)\right)
\]

with:

\[
\Sigma = D + VV^\top,\quad D = \operatorname{diag}(\sigma^2)
\]

### Efficient log-determinant

The implementation uses the matrix determinant lemma:

\[
\det(D + VV^\top) = \det(D)\det(I_k + V^\top D^{-1}V)
\]

so the expensive determinant is reduced to a `k x k` matrix instead of a
`d x d` one.

This logic now lives in `src/losses/kl.py`, not inside the model module.

### Per-image KL aggregation in this implementation

The model flattens the patch-token dimension and computes the mean token KL,
then multiplies by `num_patches` to recover a per-image scale:

\[
\mathrm{KL}_{image} \approx N \cdot \mathrm{mean}_{tokens}(\mathrm{KL}_{token})
\]

The same approach is used for `kl_per_dim`.

## Why `_cached_kl` and `_cached_kl_per_dim` exist

`GeneralizedPosteriorVAE.forward(...)` computes structured KL internally and
caches:

- `self._cached_kl`
- `self._cached_kl_per_dim`

This matters because the shared training loss operates on the generic VAE
signature `(recon, mu, log_var, z)`. For GP-VAE, the diagonal `log_var` alone
is not the true posterior covariance, so training and evaluation read the
cached structured KL values instead of recomputing a diagonal-only KL.

## Latent refinement path

After sampling, the model applies several latent-side steps before decoding.

### 1. Per-token latent normalization

Each token is normalized as:

\[
z \leftarrow \frac{z}{\operatorname{std}(z,\text{dim}=-1) + \epsilon}
\]

This helps stabilize decoding by limiting uncontrolled latent magnitude
variation.

### 2. Global token

A learnable global token is prepended:

\[
z \leftarrow [z_g; z_1; \dots; z_N]
\]

This gives the refinement stage a global coordination channel.

### 3. Latent positional embeddings

`latent_pos` is added after prepending the global token so the refinement
transformer knows where each token belongs in the patch ordering.

### 4. Latent refinement transformer

`LatentRefinement` is a small transformer encoder operating in latent space.

Its role is to:

- redistribute information across patch latents
- use the global token for long-range coordination
- improve consistency before decoding

After refinement, the global token is dropped and only patch latents are
decoded.

## Decoder pipeline

The decoder path is:

1. project latent tokens with `dec_proj`
2. run them through `transformer_decoder`
3. decode each token into a patch with `PatchTokenDecoder`
4. `unpatchify(...)` the patch tensor back into an image
5. apply seam-aware residual refinement
6. apply `sigmoid`

### PatchTokenDecoder

`PatchTokenDecoder`:

- maps a token to a small feature map through `fc`
- upsamples with `ConvTranspose2d`
- applies a final `Conv2d`
- interpolates to exact patch size if needed

So decoding is still patch-local at that stage. Global consistency is improved
by the latent transformers and the seam refiner.

## Seam-aware refinement

Patch-wise decoding can leave visible seams at patch boundaries.

To reduce those artifacts, the model creates a seam mask:

\[
m \in \{0,1\}^{1 \times 1 \times H \times W}
\]

with ones near internal patch boundaries and zeros elsewhere.

Then a small convolutional refiner predicts:

\[
\Delta = f_\theta(\hat{x})
\]

and applies it only where the seam mask is active:

\[
\hat{x} \leftarrow \hat{x} + m \odot \Delta
\]

Finally:

\[
\hat{x} \leftarrow \sigma(\hat{x})
\]

This keeps the refinement localized instead of globally altering the whole
image.

## Sampling path

The `sample(...)` method draws patch latents from a standard normal prior:

\[
z \sim \mathcal{N}(0, T^2 I)
\]

where `temperature` scales the samples.

Optional `truncation` clamps latent values elementwise:

\[
z \leftarrow \operatorname{clip}(z, -\tau, +\tau)
\]

Then sampling follows the same latent path as training:

- per-token normalization
- prepend global token
- add latent positional embeddings
- latent refinement
- drop global token
- decode

This keeps training-time and sampling-time behavior closely aligned.

## Public API of the current GP-VAE class

The current `GeneralizedPosteriorVAE` exposes:

- `encode_distribution(x) -> (mu, log_sigma, V)`
- `encode_latent_mean(x) -> mu`
- `encode(x) -> (mu, log_sigma, V)`
- `decode(z) -> image`
- `forward(x) -> (recon, mu, log_var, z)`
- `sample(n_samples, device, temperature=..., truncation=...)`
- `get_kl_override()`
- `get_kl_per_dim()`

`forward(...)` returns a `log_var` tensor for compatibility with the shared VAE
training interface, but the true GP-VAE KL comes from the cached structured
posterior quantities.

## Practical considerations

- If beta ramps too quickly, reconstructions can degrade because the posterior
  is pushed too close to the prior too early.
- If `covariance_rank` is too small, the model cannot capture enough correlated
  uncertainty; if it is too large, optimization becomes harder.
- If the seam refiner is too weak, patch boundaries remain visible; if too
  strong, transitions may become oversmoothed.
- Latent normalization improves stability but also constrains latent scale; the
  refinement transformer and global token help recover flexibility.
- Larger `patch_div` increases token count and transformer cost.

## Summary

The current GP-VAE is best understood as:

- a patch-token latent model
- with a structured posterior
- plus latent-space coordination
- and localized seam cleanup

The main difference from the plain VAE in this repo is that the posterior
covariance is not diagonal, so the KL term must be computed from the structured
posterior and fed into the generic training pipeline through the cached
override path.
