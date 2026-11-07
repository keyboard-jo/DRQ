"""Unit tests for the pure metric helpers and small tensor functions in
train_drq_vs_rqvae.py. These exercise the diagnostic math on tiny CPU inputs;
they do NOT run the training loop (no GPU needed).
"""

import argparse
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import train_drq_vs_rqvae as t

_logger = logging.getLogger("test_drq_vs_rqvae")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_gini_uniform_is_zero():
    assert t.gini_coefficient(np.array([10.0] * 5)) == pytest.approx(0.0, abs=1e-9)


def test_gini_concentrated_approaches_one():
    vals = np.array([100.0] + [0.0] * 99)
    assert t.gini_coefficient(vals) > 0.9


def test_gini_empty_and_zero():
    assert t.gini_coefficient(np.array([])) == 0.0
    assert t.gini_coefficient(np.array([0.0, 0.0])) == 0.0


def test_effective_cells_all_unique_returns_n():
    assert t.effective_cells(100, 100) == 100


def test_effective_cells_monotonic_in_uniqueness():
    assert t.effective_cells(1000, 500) > t.effective_cells(1000, 100)


def test_tuple_redundancy_zero_when_independent():
    nats, bits = t.tuple_redundancy([1.0, 2.0], 3.0)
    assert nats == pytest.approx(0.0, abs=1e-12)
    assert bits == pytest.approx(0.0, abs=1e-12)


def test_tuple_redundancy_positive_when_dependent():
    nats, bits = t.tuple_redundancy([1.0, 1.0], 1.0)
    assert nats == pytest.approx(1.0, abs=1e-12)
    assert bits == pytest.approx(math.log2(math.e), rel=1e-9)


def test_utilisation_deficit_bounds():
    assert t.utilisation_deficit(math.log(256), 256) == pytest.approx(0.0, abs=1e-9)
    assert t.utilisation_deficit(0.0, 256) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Overlap metrics (O_pi / K_eff decomposition)
# --------------------------------------------------------------------------- #
def test_overlap_uniform_separated_gives_k_eff_eq_k():
    k = 8
    counts = torch.ones(k, dtype=torch.long)
    centroids = torch.eye(k) * 10.0  # well separated on unit-scale
    om = t._overlap_metrics(counts, centroids, sigma=0.0)
    assert om["overlap_rate"] == pytest.approx(1.0 / k, abs=1e-4)
    assert om["effective_codes"] == pytest.approx(float(k), rel=1e-3)
    assert om["distribution_floor"] == pytest.approx(1.0 / k, abs=1e-9)
    assert om["geometry_cross_overlap"] == pytest.approx(0.0, abs=1e-3)


def test_overlap_collapsed_gives_k_eff_one():
    k = 8
    counts = torch.zeros(k, dtype=torch.long)
    counts[0] = 100
    centroids = torch.eye(k) * 10.0
    om = t._overlap_metrics(counts, centroids, sigma=0.0)
    assert om["overlap_rate"] == pytest.approx(1.0, abs=1e-6)
    assert om["effective_codes"] == pytest.approx(1.0, abs=1e-6)
    assert om["distribution_floor"] == pytest.approx(1.0, abs=1e-9)
    assert om["geometry_cross_overlap"] == pytest.approx(0.0, abs=1e-6)


def test_overlap_duplicate_centroids_fail_injectivity():
    """Two used codes sitting on the same point must collapse K_eff to 1.

    This is the pure-geometry failure mode: perfect marginal usage (pi=0.5/0.5)
    but zero distinguishable capacity, because the two codes are the same point.
    """
    counts = torch.tensor([50, 50], dtype=torch.long)
    centroids = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    om = t._overlap_metrics(counts, centroids, sigma=0.0)
    assert om["distribution_floor"] == pytest.approx(0.5, abs=1e-9)
    assert om["overlap_rate"] == pytest.approx(1.0, abs=1e-7)
    assert om["effective_codes"] == pytest.approx(1.0, abs=1e-6)
    assert om["geometry_cross_overlap"] == pytest.approx(0.5, abs=1e-7)


def test_auto_sigma_scales_with_centroid_spread():
    base = torch.eye(6) * 1.0
    wide = torch.eye(6) * 100.0
    assert t.auto_sigma(wide) > t.auto_sigma(base)


def test_auto_sigma_is_a_fraction_of_nearest_neighbour_distance():
    c = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    assert t.auto_sigma(c) == pytest.approx(t.SIGMA_TOL * 10.0, rel=1e-6)


def test_auto_sigma_ignores_unused_codes():
    centroids = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1e6, 1e6]])
    counts = torch.tensor([50, 50, 0], dtype=torch.long)
    s = t.auto_sigma(centroids, counts)
    assert s == pytest.approx(t.SIGMA_TOL * 1.0, rel=1e-6)


def test_k_eff_equals_participation_ratio_when_codes_are_spread():
    """Well-separated codes: geometry adds ~nothing, K_eff = 1/sum(pi^2)."""
    counts = torch.ones(16, dtype=torch.long)
    centroids = torch.randn(16, 64) * 5.0
    om = t._overlap_metrics(counts, centroids, sigma=0.0)
    assert om["effective_codes"] == pytest.approx(om["participation_ratio"], rel=1e-3)
    assert om["effective_codes"] == pytest.approx(16.0, rel=1e-3)
    assert om["geometry_cross_overlap"] == pytest.approx(0.0, abs=1e-3)


def test_k_eff_drops_below_participation_ratio_for_duplicate_codes():
    """The key point: perfect marginal usage, but two codes are the same point.

    Participation ratio is blind to this (says 2), K_eff is not (says 1).
    """
    centroids = torch.randn(16, 64) * 5.0
    centroids[1] = centroids[0]
    counts = torch.zeros(16, dtype=torch.long)
    counts[0], counts[1] = 50, 50
    om = t._overlap_metrics(counts, centroids, sigma=0.0)
    assert om["participation_ratio"] == pytest.approx(2.0, rel=1e-6)
    assert om["effective_codes"] == pytest.approx(1.0, abs=1e-3)
    assert om["geometry_cross_overlap"] == pytest.approx(0.5, abs=1e-3)


def test_k_eff_tracks_skewed_usage_over_spread_codes():
    counts = torch.tensor([100, 50, 25, 12, 6, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                          dtype=torch.long)
    centroids = torch.randn(16, 64) * 5.0
    om = t._overlap_metrics(counts, centroids, sigma=0.0)
    pr = float((counts.float().sum() ** 2) / (counts.float() ** 2).sum())
    assert om["participation_ratio"] == pytest.approx(pr, rel=1e-5)
    assert om["effective_codes"] == pytest.approx(pr, rel=1e-2)


# --------------------------------------------------------------------------- #
# Assignment / k-means
# --------------------------------------------------------------------------- #
def test_assign_nearest():
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    c = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = t.assign_nearest(x, c, batch_size=2)
    assert torch.equal(labels, torch.tensor([0, 1, 2]))


def test_hierarchical_kmeans_runs_and_shapes():
    torch.manual_seed(0)
    np.random.seed(0)
    # 4 well-separated Gaussian clusters -> 100 x 4
    z = torch.cat([torch.randn(25, 4) + i for i in [0.0, 10.0, 20.0, 30.0]], dim=0)
    centroids, idx = t.hierarchical_kmeans(
        z, [4, 4], _logger, fit_samples=100, n_init=2, niter=20,
        backend="auto", prefer_gpu=False, random_state=0, assign_batch_size=64,
    )
    assert len(centroids) == 2
    assert [c.shape for c in centroids] == [(4, 4), (4, 4)]
    assert idx.shape == (100, 2)
    assert idx.min().item() >= 0 and idx.max().item() < 4
    # First level should recover the 4 planted clusters
    assert torch.unique(idx[:, 0]).numel() == 4


def test_hierarchical_kmeans_residual_reduces_error():
    torch.manual_seed(0)
    z = torch.randn(200, 4) * 3.0
    centroids, idx = t.hierarchical_kmeans(
        z, [8, 8], _logger, fit_samples=200, n_init=2, niter=20,
        backend="auto", prefer_gpu=False, random_state=0, assign_batch_size=64,
    )
    err1 = (z - centroids[0][idx[:, 0]]).pow(2).sum(-1).mean()
    err2 = (z - (centroids[0][idx[:, 0]] + centroids[1][idx[:, 1]])).pow(2).sum(-1).mean()
    assert err2 < err1


def test_hierarchical_kmeans_fit_pool_limits_fitting():
    torch.manual_seed(0)
    z = torch.randn(40, 3)
    pool = z[:10]
    centroids, idx = t.hierarchical_kmeans(
        z, [2], _logger, fit_samples=10, n_init=1, niter=10,
        backend="sklearn", prefer_gpu=False, random_state=0,
        assign_batch_size=8, fit_pool=pool,
    )
    assert idx.shape == (40, 1)
    assert centroids[0].shape == (2, 3)


# --------------------------------------------------------------------------- #
# Tuple / stage diagnostics
# --------------------------------------------------------------------------- #
def test_tuple_diagnostics_collision():
    idx = torch.tensor([[0, 0], [0, 0], [1, 1], [2, 2]])
    d = t.compute_tuple_diagnostics(idx)
    assert d["n_items"] == 4
    assert d["n_unique_tuples"] == 3
    assert d["uniqueness_ratio"] == pytest.approx(0.75)
    # DRQ definition: 1 - unique/N
    assert d["collision_rate"] == pytest.approx(0.25)
    # items sitting in a bucket of size >= 2
    assert d["items_in_collision_frac"] == pytest.approx(0.5)
    assert d["items_in_collision"] == 2
    assert d["n_colliding_tuples"] == 1
    assert d["max_collision_bucket"] == 2


def test_collision_rate_matches_drq_paper_numbers():
    """Reproduce the paper's arithmetic: 1 - 385,551/500,000 = 0.228898."""
    n, n_unique = 500_000, 385_551
    others = n - n_unique  # items beyond the first in each bucket
    idx = torch.zeros(n, 1, dtype=torch.long)
    # n_unique distinct ids, then pile the excess onto the first one
    idx[:n_unique, 0] = torch.arange(n_unique)
    idx[n_unique:, 0] = 0
    d = t.compute_tuple_diagnostics(idx)
    assert others == 114_449
    assert d["n_unique_tuples"] == n_unique
    assert d["collision_rate"] == pytest.approx(0.228898, abs=1e-6)


def test_tuple_diagnostics_no_collision():
    idx = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
    d = t.compute_tuple_diagnostics(idx)
    assert d["n_unique_tuples"] == 4
    assert d["collision_rate"] == pytest.approx(0.0)
    assert d["max_collision_bucket"] == 1


def test_stage_diagnostics():
    idx = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    codebooks = [torch.eye(2), torch.eye(2)]
    out = t.compute_stage_diagnostics(idx, codebooks, sigma=0.1)
    assert len(out) == 2
    for m in out:
        assert m["used_codes"] == 2
        assert m["total_codes"] == 2
        assert m["dead_code_pct"] == pytest.approx(0.0, abs=1e-6)
        assert m["max_freq"] == pytest.approx(0.5)
        assert m["perplexity"] == pytest.approx(2.0, rel=1e-4)
        assert "gini" in m and "effective_codes" in m and "overlap_rate" in m
        assert m["kernel_sigma"] > 0


def test_prefix_uniqueness():
    idx = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    ratios = t.compute_prefix_uniqueness(idx)
    assert ratios[0] == pytest.approx(0.5)
    assert ratios[1] == pytest.approx(1.0)


def test_run_diagnostics_end_to_end():
    idx = torch.tensor([[0, 0], [0, 0], [1, 1], [2, 2]])
    codebooks = [torch.eye(3), torch.eye(3)]
    d = t.run_diagnostics(idx, codebooks, sigma=0.1)
    assert "stage" in d and "tuples" in d and "prefix_uniqueness" in d
    assert d["tuples"]["n_unique_tuples"] == 3
    assert "redundancy_nats" in d and "redundancy_bits" in d


# --------------------------------------------------------------------------- #
# Models (tiny, CPU)
# --------------------------------------------------------------------------- #
def test_continuous_vae_shapes():
    vae = t.ContinuousVAE(16, 8)
    x = torch.randn(3, 16)
    z, x_hat = vae(x)
    assert z.shape == (3, 8)
    assert x_hat.shape == (3, 16)


def test_residual_quantizer_forward_encode_consistent():
    torch.manual_seed(0)
    q = t.ResidualQuantizer([4, 4], 8)
    z = torch.randn(6, 8)
    out = q(z)
    assert out["indices"].shape == (6, 2)
    recon = sum(q.codebooks[d][out["indices"][:, d]] for d in range(2))
    assert torch.allclose(out["quantized_sum"], recon, atol=1e-6)
    assert torch.equal(out["indices"], q.encode_indices(z))


def test_rqvae_loss_components():
    torch.manual_seed(0)
    m = t.RQVAE(16, 8, [4, 4], beta=0.25)
    x = torch.randn(3, 16)
    out = m(x)
    assert out["indices"].shape == (3, 2)
    assert out["loss"].ndim == 0
    assert out["recon_loss"].ndim == 0
    expected = out["recon_loss"] + out["cb_loss"] + 0.25 * out["cm_loss"]
    assert out["loss"].item() == pytest.approx(expected.item(), rel=1e-5)


# --------------------------------------------------------------------------- #
# Loss definitions (paper-faithful squared L2) and stage-1 contrastive term
# --------------------------------------------------------------------------- #
def test_sq_err_is_sum_over_dims_mean_over_batch():
    a = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    b = torch.zeros(2, 2)
    # row 0: 1+4=5 ; row 1: 0  ->  mean = 2.5
    assert t.sq_err(a, b).item() == pytest.approx(2.5)


def test_info_nce_low_for_aligned_pairs_high_for_random():
    torch.manual_seed(0)
    za = torch.randn(64, 16)
    aligned = t.info_nce(za, za + 1e-3 * torch.randn_like(za), tau=0.1).item()
    random = t.info_nce(za, torch.randn(64, 16), tau=0.1).item()
    assert aligned < random
    assert aligned < 0.1


def test_load_positive_pairs_maps_ids_and_drops_unknown(tmp_path):
    import pandas as pd
    p = tmp_path / "pairs.csv"
    pd.DataFrame({"item_a": ["a", "c", "zzz"], "item_b": ["b", "d", "a"]}).to_csv(p, index=False)
    pairs = t.load_positive_pairs(str(p), ["a", "b", "c", "d"], _logger)
    assert pairs.tolist() == [[0, 1], [2, 3]]


def test_stage1_lambda_zero_is_reconstruction_only():
    torch.manual_seed(0)
    emb = torch.randn(64, 16)
    loader = DataLoader(emb, batch_size=16, shuffle=False)
    vae = t.ContinuousVAE(16, 8)
    t.train_continuous_vae(vae, loader, loader, epochs=1, lr=1e-3, warmup_steps=1,
                           min_lr_ratio=0.1, device=torch.device("cpu"), logger=_logger,
                           lam=0.0)


def test_stage1_lambda_positive_requires_pairs():
    emb = torch.randn(16, 8)
    loader = DataLoader(emb, batch_size=8)
    vae = t.ContinuousVAE(8, 4)
    with pytest.raises(ValueError):
        t.train_continuous_vae(vae, loader, loader, epochs=1, lr=1e-3, warmup_steps=1,
                               min_lr_ratio=0.1, device=torch.device("cpu"),
                               logger=_logger, lam=0.3)


def test_stage1_lambda_positive_runs_with_pairs():
    torch.manual_seed(0)
    emb = torch.randn(32, 8)
    loader = DataLoader(emb, batch_size=8)
    pairs = torch.tensor([[i, (i + 1) % 32] for i in range(32)])
    vae = t.ContinuousVAE(8, 4)
    t.train_continuous_vae(vae, loader, loader, epochs=1, lr=1e-3, warmup_steps=1,
                           min_lr_ratio=0.1, device=torch.device("cpu"), logger=_logger,
                           embeddings=emb, pairs=pairs, lam=0.3, tau=0.1, pair_batch=8)


# --------------------------------------------------------------------------- #
# Dead-code revival (RQ-VAE baseline arm)
# --------------------------------------------------------------------------- #
def test_rqvae_forward_residuals_are_stage_inputs():
    torch.manual_seed(0)
    m = t.RQVAE(input_dim=16, hidden_dim=8, codebook_sizes=[4, 4, 4], beta=0.25)
    out = m(torch.randn(10, 16), return_residuals=True)
    assert len(out["residuals"]) == 4          # z, then after each of the 3 stages
    assert out["residuals"][0].shape == (10, 8)
    assert torch.allclose(out["residuals"][1], out["residuals"][0] - m.quantizer.codebooks[0][out["indices"][:, 0]].detach())


def test_reviver_replaces_dead_codes_and_resets_adam_state():
    torch.manual_seed(0)
    cb = torch.nn.Parameter(torch.zeros(4, 3))
    opt = torch.optim.AdamW([cb], lr=1e-3)
    cb.grad = torch.ones_like(cb)
    opt.step()                                  # populate exp_avg / exp_avg_sq
    rev = t.DeadCodeReviver([4], hidden_dim=3, device=torch.device("cpu"), threshold_frac=0.025)
    res = [torch.full((50, 3), 7.0)]
    idx = torch.tensor([[0]] * 40 + [[1]] * 10)  # codes 2 and 3 never used
    rev.update(res, idx)
    n = rev.revive([cb], opt)
    assert n == 2
    assert torch.allclose(cb.data[2], torch.full((3,), 7.0))
    assert torch.allclose(cb.data[3], torch.full((3,), 7.0))
    assert torch.all(opt.state[cb]["exp_avg"][2] == 0) and torch.all(opt.state[cb]["exp_avg_sq"][3] == 0)
    assert torch.all(opt.state[cb]["exp_avg"][0] != 0)   # used codes keep their moments


def test_reviver_ring_buffer_wraps():
    rev = t.DeadCodeReviver([2], hidden_dim=1, device=torch.device("cpu"), buffer_size=4)
    idx = torch.zeros(3, 1, dtype=torch.long)
    rev.update([torch.tensor([[1.], [2.], [3.]])], idx)     # ptr 0 -> 3
    assert rev.ptrs[0] == 3 and not rev.has_wrapped[0]
    rev.update([torch.tensor([[4.], [5.], [6.]])], idx)     # wraps: slots 3,0,1
    assert rev.ptrs[0] == 2 and rev.has_wrapped[0]
    assert rev.buffers[0].flatten().tolist() == [5.0, 6.0, 3.0, 4.0]


def test_train_rqvae_runs_with_and_without_revival():
    torch.manual_seed(0)
    emb = torch.randn(64, 16)
    loader = DataLoader(emb, batch_size=16, shuffle=False)
    for fix in (True, False):
        m = t.RQVAE(input_dim=16, hidden_dim=8, codebook_sizes=[8, 8], beta=0.25)
        t.train_rqvae(m, loader, loader, epochs=2, lr=1e-3, warmup_steps=1, min_lr_ratio=0.1,
                      device=torch.device("cpu"), logger=_logger,
                      dead_code_fix=fix, dead_code_interval=2)


def test_default_beta_and_ae_warmup_off():
    assert t.BETA == 0.1
    assert t.AE_WARMUP_EPOCHS == 0


def test_train_rqvae_with_ae_warmup_runs():
    torch.manual_seed(0)
    emb = torch.randn(64, 16)
    loader = DataLoader(emb, batch_size=16, shuffle=False)
    m = t.RQVAE(input_dim=16, hidden_dim=8, codebook_sizes=[8, 8], beta=0.1)
    t.train_rqvae(m, loader, loader, epochs=1, lr=1e-3, warmup_steps=1, min_lr_ratio=0.1,
                  device=torch.device("cpu"), logger=_logger, ae_warmup_epochs=1)

def test_sigma_scan_spread_codebook_holds_k_eff():
    torch.manual_seed(0)
    k = 64
    centroids = torch.randn(k, 4)
    counts = torch.full((k,), 10)
    base = t.auto_sigma(centroids, counts)
    scan = t.sigma_scan(counts, centroids, base, [0.25, 0.5, 1.0, 2.0])
    assert [s["sigma_mult"] for s in scan] == [0.25, 0.5, 1.0, 2.0]
    assert all(s["sigma"] == base * s["sigma_mult"] for s in scan)
    keff = [s["effective_codes"] for s in scan]
    # spread codes: K_eff stays close to the participation ratio (=k here)
    assert keff[0] > 0.9 * k
    assert keff[-1] <= keff[0]  # monotone: wider kernel, more overlap


def test_sigma_scan_colliding_codes_collapse_first():
    # two duplicate centroids: their pair mass merges already at tiny sigma
    centroids = torch.zeros(2, 3)
    centroids[1] = torch.tensor([1e-3, 0.0, 0.0])
    counts = torch.tensor([50, 50])
    scan = t.sigma_scan(counts, centroids, 1e-4, [0.25, 4.0])
    assert scan[0]["effective_codes"] > scan[1]["effective_codes"]


def test_run_diagnostics_includes_scan_and_stays_metric_only():
    torch.manual_seed(0)
    codebooks = [torch.randn(16, 4)]
    indices = torch.randint(0, 16, (200, 1))
    diag = t.run_diagnostics(indices, codebooks, 0.0, [0.5, 1.0, 2.0])
    assert "sigma_scan" in diag["stage"][0]
    assert len(diag["stage"][0]["sigma_scan"]) == 3
    # scan must not alter the headline fields computed at base sigma
    plain = t.run_diagnostics(indices, codebooks, 0.0, None)
    for key in ("overlap_rate", "effective_codes", "kernel_sigma"):
        assert diag["stage"][0][key] == plain["stage"][0][key]


# --------------------------------------------------------------------------- #
# Warmup arm (AE warmup -> joint RQ-VAE)
# --------------------------------------------------------------------------- #
def test_warmup_arm_constants_and_cli():
    assert t.WARMUP_ARM_EPOCHS == 5
    assert t.HANDOFF_WARMUP_STEPS == 200


def test_joint_warmup_steps_handoff_shortening():
    # no AE warmup: full global ramp
    assert t._joint_warmup_steps(2000, 0) == 2000
    # after AE warmup: ramp capped at the handoff value
    assert t._joint_warmup_steps(2000, 5) == 200
    # global ramp shorter than the cap stays as-is
    assert t._joint_warmup_steps(100, 5) == 100


class _FakeDataset:
    """Minimal stand-in for EmbeddingDataset (torch tensor + ids, indexable)."""
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings
        self.item_ids = [str(i) for i in range(embeddings.shape[0])]

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx]


def _warmup_arm_args():
    return argparse.Namespace(
        input_dim=16, hidden_dim=8, beta=0.1, epochs=1, lr=1e-3,
        warmup_steps=2, min_lr_ratio=0.1, batch_size=8,
        inference_batch_size=16, dead_code_fix=True,
        dead_code_fix_interval=500, dead_code_threshold_frac=0.1,
        usage_decay=0.99, ae_warmup_epochs=0,
    )


def test_warmup_arm_end_to_end_small():
    torch.manual_seed(0)
    ds = _FakeDataset(torch.randn(32, 16))
    idx = np.arange(32)
    res = t.run_rqvae_arm(ds, idx[:24], idx[24:], [8, 8], _warmup_arm_args(), _logger,
                          name="warmup", ae_warmup_epochs=1)
    assert res["indices"].shape == (32, 2)
    assert len(res["codebooks"]) == 2 and res["codebooks"][0].shape == (8, 8)
    assert res["recon_continuous"] > 0 and res["recon_quantized"] > 0


def test_warmup_arm_codebooks_stay_trunc_normal_init_not_kmeans():
    # the warmup arm must introduce a fresh (trunc-normal) codebook at handoff,
    # not k-means centroids: handoff codebook == quantizer codebooks
    torch.manual_seed(0)
    ds = _FakeDataset(torch.randn(32, 16))
    idx = np.arange(32)
    res = t.run_rqvae_arm(ds, idx[:24], idx[24:], [8, 8], _warmup_arm_args(), _logger,
                          name="warmup", ae_warmup_epochs=1)
    for cb_c, cb_m in zip(res["codebooks"], res["model"].quantizer.codebooks):
        assert torch.allclose(cb_c, cb_m.detach().cpu())


def test_build_comparison_generic_baseline_candidate_keys():
    def fake_diag(perp):
        return {
            "tuples": {"n_unique_tuples": 100, "uniqueness_ratio": 0.5,
                       "collision_rate": 0.5, "max_collision_bucket": 3,
                       "effective_cells": 90, "joint_perplexity": perp},
            "stage": [{"perplexity": perp, "gini": 0.2, "effective_codes": 50.0,
                       "overlap_rate": 0.02}],
        }
    cmp = t.build_comparison(fake_diag(10.0), fake_diag(20.0))
    assert set(cmp) == {"baseline", "candidate"}
    assert cmp["baseline"]["stage_headline"]["level0_perplexity"] == 10.0
    assert cmp["candidate"]["tuples"]["joint_perplexity"] == 20.0
