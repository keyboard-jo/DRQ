# References

Ordered by relevance to **injectivity / collision** in RQ-VAE semantic IDs.

## 1. DRQ — Decoupled Residual Quantization for Robust Semantic IDs (arXiv:2606.01844)

Shopee, June 2026. **Primary reference.** Develops the diagnostic framework this
repo adopts: *expected codeword overlap* (`O_π`) and *effective codebook size*
(`K_eff = 1/O_π`), decomposing SID degradation into a *distribution penalty*
(self-collision floor `Σ π_i²`) and a *geometry penalty* (Minkowski-sum codebook
on a curved manifold). Proposes Decoupled Residual Quantization (DRQ): train a
continuous VAE (no STE / commitment loss), then post-hoc hierarchical K-Means.
Cuts collision rate 0.229 → 0.075 vs RQ-VAE on a 15M-item industrial set.

## 2. QuaSID — Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning (arXiv:2603.00632)

Tencent / Kuaishou, KDD 2026. Qualification-aware SID learning: not all
collisions are equally harmful; treats them non-uniformly. +5.9% top-ranking,
increases SID diversity, live A/B on Kuaishou e-commerce.

## 3. FORGE — Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets (arXiv:2509.20904)

Sept 2025. Post-hoc collision resolution: KNN-based and random-based
re-assignment for colliding items.

## 4. A Systematic Study of Semantic ID Spaces for Generative Information Retrieval (arXiv:2610.08732)

Defines the *Uniqueness Ratio* `𝒰` = unique DocIDs / N and shows collision
"mechanically caps maximum achievable recall."

## 5. GenRec — RQ-VAE model notes (phonism.github.io/genrec/en/models/rqvae/)

Operational guidance: "Track layer usage, dead codes, complete-tuple collision
rate, and maximum collision group separately. Perfect usage at each layer can
coexist with many duplicate full SIDs."

## 6. GRID — Generative Recommendation with Semantic IDs: A Practitioner's Handbook (arXiv:2507.22224)

Snap, 2025. Open-source framework (RK-Means, R-VQ, RQ-VAE). Finds RK-Means and
R-VQ often beat RQ-VAE on recommendation despite 5× more RQ-VAE training.

## 7. TIGER — Recommender Systems with Generative Retrieval (arXiv:2305.05065)

Google DeepMind, NeurIPS 2023. Origin of RQ-VAE semantic IDs; appends a 4th
token purely for collision resolution.

## 8. VQ4Rec — Vector Quantization for Recommender Systems: A Review and Outlook (arXiv:2405.03110)

Survey. §6.1 frames codebook collapse as the central open problem; contextualizes
the marginal-utilization fixes already in the field.
