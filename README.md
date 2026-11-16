# RQ-VAE Semantic ID Injectivity (ONGOING REPO)

This repo investigates a single, well-defined failure of RQ-VAE semantic IDs:
**collisions** — multiple distinct items mapping to the same SID, which breaks
injectivity and mechanically caps downstream recall.

## The problem

A 3-level × 1024-codebook SID has nominal capacity 1024³ ≈ 1.07B distinct
tuples. It should injectively cover 4.8M items with room to spare. In practice
it does not: the tokenizer under-uses its codebook and concentrates items onto a
small fraction of the tuple space, so the *effective* capacity collapses far
below 1024³ and collisions appear.

Root causes (from DRQ, arXiv:2606.01844):

- **Distribution penalty** — codebook collapse skews usage, so most items pile
  onto few codes (self-collision floor `Σ π_i²`).
- **Geometry penalty** — RQ-VAE's codebook is a *Minkowski sum* (flat Euclidean
  tiling) imposed on a curved/anisotropic manifold; most code combinations sit
  where no real items live.

The four improvement arms already tested here (bottleneck norm, entropy deficit,
latent jitter, codebook LR multiplier) target *marginal* per-level utilization.
Injectivity is governed by the *joint tuple* distribution + geometry, so
"perfect usage at each layer can coexist with many duplicate full SIDs."
Marginal-utilization fixes are therefore insufficient — this is the central
finding, not a failed experiment.

## Approach

1. **Diagnose** effective codebook capacity:
   - effective cells via birthday inversion,
   - tuple total correlation,
   - per-level active codes / perplexity / Gini,
   - full-tuple collision rate + unique-ID count + max collision bucket.
2. **Fix** by decoupling continuous geometry learning from discrete distribution
   matching (DRQ-style), so geometry can spread and assignment can flatten.

## Arms (`train_drq_vs_rqvae.py`)

- `rqvae` — joint baseline (RQ-VAE, STE, dead-code revival on, trunc-normal init).
- `warmup` — 5 reconstruction-only epochs, then joint training with a cosine
  restart (200-step handoff ramp). Isolates the latent-geometry phase as the
  mechanism: sits between `rqvae` and `drq` on the coupling axis.
- `drq` — separate training: continuous VAE, then hierarchical k-means on
  frozen latents.

Run one with `--arm all` (or per arm with the same `--run_name`); comparisons
against the `rqvae` baseline are written to `comparison*.json`.

## Layout

- `train_drq_vs_rqvae.py` — the three-arm injectivity experiment.
- `references/rqvae_grid.py` — ablation-arm presets + diagnostics
  (`effective_cells`, `tuple_redundancy`, `utilisation_deficit`).
- `references/train_rqvae_my_implementation.py` — self-contained RQ-VAE
  implementation (encoder → quantizer → decoder).
- `references/train_rqvae_official_implementation.py` — upstream TIGER
  reference implementation (Gumbel-Softmax).

## Main References
- 1. DRQ — Decoupled Residual Quantization for Robust Semantic IDs (arXiv:2606.01844)
- 2. TIGER — Recommender Systems with Generative Retrieval (arXiv:2305.05065)
