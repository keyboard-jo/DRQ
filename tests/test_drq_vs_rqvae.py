"""Unit tests for the pure metric helpers and small tensor functions in
train_drq_vs_rqvae.py. These exercise the diagnostic math on tiny CPU inputs;
they do NOT run the training loop (no GPU needed).
"""

import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

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
    centroids = torch.eye(k)  # orthonormal -> well separated
    o_pi, k_eff, dist_floor, geom = t._overlap_metrics(counts, centroids, sigma=0.1)
    assert o_pi == pytest.approx(1.0 / k, abs=1e-4)
    assert k_eff == pytest.approx(float(k), rel=1e-3)
    assert dist_floor == pytest.approx(1.0 / k, abs=1e-9)
    assert geom == pytest.approx(0.0, abs=1e-6)


def test_overlap_collapsed_gives_k_eff_one():
    k = 8
    counts = torch.zeros(k, dtype=torch.long)
    counts[0] = 100
    centroids = torch.eye(k)
    o_pi, k_eff, dist_floor, geom = t._overlap_metrics(counts, centroids, sigma=0.1)
    assert o_pi == pytest.approx(1.0, abs=1e-6)
    assert k_eff == pytest.approx(1.0, abs=1e-6)
    assert dist_floor == pytest.approx(1.0, abs=1e-9)
    assert geom == pytest.approx(0.0, abs=1e-9)


def test_overlap_sigma_zero_falls_back_to_distribution_floor():
    k = 4
    counts = torch.ones(k, dtype=torch.long)
    centroids = torch.eye(k)
    o_pi, k_eff, dist_floor, geom = t._overlap_metrics(counts, centroids, sigma=0.0)
    assert o_pi == pytest.approx(dist_floor, abs=1e-9)
    assert k_eff == pytest.approx(float(k), rel=1e-6)
    assert geom == pytest.approx(0.0, abs=1e-9)


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
        z, [4, 4], _logger, fit_samples=100, n_init=3,
        random_state=0, assign_batch_size=64,
    )
    assert len(centroids) == 2
    assert [c.shape for c in centroids] == [(4, 4), (4, 4)]
    assert idx.shape == (100, 2)
    assert idx.min().item() >= 0 and idx.max().item() < 4


# --------------------------------------------------------------------------- #
# Tuple / stage diagnostics
# --------------------------------------------------------------------------- #
def test_tuple_diagnostics_collision():
    idx = torch.tensor([[0, 0], [0, 0], [1, 1], [2, 2]])
    d = t.compute_tuple_diagnostics(idx)
    assert d["n_items"] == 4
    assert d["n_unique_tuples"] == 3
    assert d["uniqueness_ratio"] == pytest.approx(0.75)
    assert d["collision_rate"] == pytest.approx(0.5)
    assert d["items_in_collision"] == 2
    assert d["n_colliding_tuples"] == 1
    assert d["max_collision_bucket"] == 2


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
