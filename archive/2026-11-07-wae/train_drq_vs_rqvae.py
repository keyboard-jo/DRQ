"""DRQ vs RQ-VAE: coupled versus decoupled residual quantization for semantic IDs.

Two arms, nothing else:

  arm 1  RQ-VAE (baseline)   geometry (encoder/decoder) and the residual
                             codebooks are trained jointly through the
                             straight-through estimator (STE).

  arm 2  DRQ                 geometry is trained first as a plain continuous
                             VAE (reconstruction only: no codebook, no STE,
                             no commitment loss), then codes are assigned
                             post-hoc with hierarchical K-Means on the frozen
                             latents.

Every mechanism from the earlier grid is stripped on purpose: no LayerNorm
bottleneck, no latent jitter, no entropy regulariser, no codebook learning-rate
multiplier, no K-Means codebook initialisation, and no dead-code revival. The
only difference between the arms is *when* the discrete assignment happens.

Injectivity diagnostics follow DRQ (arXiv:2606.01844):

  per level   active codes, perplexity, Gini, distribution floor sum(pi^2),
              expected overlap O_pi, effective codebook size K_eff = 1/O_pi,
              geometry cross-overlap (O_pi minus the distribution floor).
  full tuple  unique tuples, collision rate = 1 - unique/N (the definition
              used in the DRQ paper), fraction of items in a non-singleton
              bucket, max collision bucket, joint entropy/perplexity,
              effective cells (birthday inversion), stage redundancy.

The Gaussian kernel width sigma defaults to auto (per-level median pairwise
centroid distance). A fixed small sigma makes the geometry term vanish, since
exp(-d^2 / 4sigma^2) -> 0 for well-separated codes, silently collapsing K_eff
to the participation ratio 1/sum(pi^2).

Run one arm:

  python train_drq_vs_rqvae.py --arm rqvae --input_path data/mm_emb_82.parquet
  python train_drq_vs_rqvae.py --arm drq   --input_path data/mm_emb_82.parquet

Run both and compare (default):

  python train_drq_vs_rqvae.py --input_path data/mm_emb_82.parquet

No GPU is required by this script itself, but training is slow on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------- #
# Constants (shared training recipe, identical for both arms)
# --------------------------------------------------------------------------- #
DEFAULT_INPUT_PATH = "tencent_gr_parquet/train/mm_emb_82.parquet"
DEFAULT_ITEM_ID_COL = "item_id"
DEFAULT_EMBEDDING_COL = "embedding"

RANDOM_SEED = 42
INPUT_DIM = 256
HIDDEN_DIM = 64
CODEBOOK_SIZE = 1024      # user's real target: 3 x 1024 for 4.8M items
NUM_CODEBOOKS = 3
BETA = 0.1                # commitment weight (matches references/train_rqvae_my_implementation.py default)
LR = 5e-4
EPOCHS = 50
BATCH_SIZE = 4096
INFERENCE_BATCH_SIZE = 16384
VAL_RATIO = 0.1
WARMUP_STEPS = 2000
MIN_LR_RATIO = 0.01
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
# 0.0 -> auto (calibrated to the centroid spacing; see auto_sigma). Do not
# hardcode a small absolute value: the kernel exp(-d^2/4sigma^2) then collapses
# to the identity and K_eff degenerates to 1/sum(pi^2), which hides exactly the
# geometry failure mode DRQ is about.
OVERLAP_SIGMA = 0.0
# auto sigma tolerance: fraction of the median nearest-neighbour centroid distance
SIGMA_TOL = 0.1
KMEANS_FIT_SAMPLES = 1_000_000
# DRQ stage 1 contrastive weight. 0.0 = paper's DRQ-VAE; 0.3 = DRQ-VAE+CL.
DRQ_LAMBDA = 0.0
DRQ_TAU = 0.1            # InfoNCE temperature: not stated in the paper we have; assumed
DRQ_PAIR_BATCH = 1024
KMEANS_N_INIT = 3
KMEANS_NITER = 50
# Metric-only sigma sweep (see sigma_scan): K_eff evaluated at mult * auto_sigma
# per level. The two arms are compared as curves, not at one arbitrary sigma.
SIGMA_SCAN_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]
# Dead-code revival (ported from references/train_rqvae_my_implementation.py, GPURadialResidualFixer).
# RQ-VAE baseline arm only; the DRQ arm has no codebook in stage 1.
DEAD_CODE_FIX = True
# Optional reconstruction-only pre-training of encoder+decoder before joint training (RQ-VAE arm).
# Not part of the RQ-VAE paper; off by default. Kept as a flag for ablation.
AE_WARMUP_EPOCHS = 0
# Third arm (--arm warmup): AE warmup as an explicit ablation point.
# 5 reconstruction-only epochs, then joint RQ-VAE training for --epochs (total = 5 + --epochs).
WARMUP_ARM_EPOCHS = 5
# LR ramp after the warmup->joint handoff when the cosine schedule restarts.
HANDOFF_WARMUP_STEPS = 200
DEAD_CODE_FIX_INTERVAL = 500            # steps between revivals
DEAD_CODE_THRESHOLD_FRAC = 0.1          # dead = EMA usage < 10% of mean code usage
DEAD_CODE_REVIVE_UNTIL_FRAC = 0.6       # no revival after 60% of total steps
USAGE_DECAY = 0.99                      # EMA decay for per-code usage
BUFFER_CAPACITY_PER_STAGE = 20_000      # ring buffer of recent stage inputs per level


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("DRQ_RQVAE")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def log_table(logger: logging.Logger, rows: list[tuple[str, str, str]],
              col_names: tuple[str, str, str] = ("metric", "value", "extra")) -> None:
    if not rows:
        return
    c0 = max(len(r[0]) for r in rows)
    c1 = max(len(r[1]) for r in rows)
    c2 = max(len(r[2]) for r in rows)
    c0 = max(c0, len(col_names[0]))
    c1 = max(c1, len(col_names[1]))
    c2 = max(c2, len(col_names[2]))
    sep = f"+-{'-' * c0}-+-{'-' * c1}-+-{'-' * c2}-+"
    logger.info(sep)
    logger.info(f"| {col_names[0]:<{c0}} | {col_names[1]:<{c1}} | {col_names[2]:<{c2}} |")
    logger.info(sep)
    for a, b, c in rows:
        logger.info(f"| {a:<{c0}} | {b:<{c1}} | {c:<{c2}} |")
    logger.info(sep)


# --------------------------------------------------------------------------- #
# Data loader (parquet + h5). Keep as-is; parquet is the primary path.
# --------------------------------------------------------------------------- #
class EmbeddingDataset(Dataset):
    """Loads precomputed item embeddings (parquet or h5)."""

    def __init__(self, input_file_path: str, max_samples: int | None = None,
                 logger: logging.Logger | None = None) -> None:
        self.input_file_path = input_file_path
        path = input_file_path
        log = logger.info if logger else print
        t0 = time.time()

        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            log(f"Loading parquet: {path}")
            table = pq.read_table(
                path, columns=[DEFAULT_ITEM_ID_COL, DEFAULT_EMBEDDING_COL]
            )
            if max_samples is not None:
                table = table.slice(0, max_samples)
            item_ids_raw = table[DEFAULT_ITEM_ID_COL].to_pandas().values
            embeddings_raw = np.stack(table[DEFAULT_EMBEDDING_COL].to_numpy())
            self.item_ids = [str(o) for o in item_ids_raw]
            self.embeddings = torch.from_numpy(embeddings_raw.astype(np.float32))
        elif path.endswith((".h5", ".hdf5")):
            import h5py
            log(f"Loading h5: {path}")
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                emb_key = DEFAULT_EMBEDDING_COL if DEFAULT_EMBEDDING_COL in keys else keys[0]
                id_key = DEFAULT_ITEM_ID_COL if DEFAULT_ITEM_ID_COL in keys else None
                dset = f[emb_key]
                n = dset.shape[0]
                if max_samples is not None:
                    n = min(n, max_samples)
                self.embeddings = torch.from_numpy(dset[:n].astype(np.float32))
                if id_key is not None:
                    self.item_ids = [str(o) for o in f[id_key][:n]]
                else:
                    self.item_ids = [str(i) for i in range(n)]
        else:
            raise ValueError(f"Unsupported file type: {path}")

        log(f"  loaded {len(self.item_ids)} items, embedding dim "
            f"{self.embeddings.shape[1] if self.embeddings.ndim > 1 else '?'} "
            f"({time.time() - t0:.2f}s)")

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


def sq_err(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared L2 norm per row, averaged over the batch: mean_n ||a_n - b_n||^2.

    Matches the paper's ||.||_2^2 (sum over latent/embedding dims, mean over items).
    """
    return ((a - b) ** 2).sum(dim=-1).mean()


def info_nce(za: torch.Tensor, zb: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of positive pairs (za_i, zb_i).

    Cosine similarity / tau, in-batch negatives. Positive pairs come from the
    user-supplied pair file (co-viewed or semantically similar items).
    """
    za = F.normalize(za, dim=-1)
    zb = F.normalize(zb, dim=-1)
    logits = za @ zb.t() / tau
    labels = torch.arange(za.shape[0], device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def load_positive_pairs(path: str, item_ids: list[str], logger: logging.Logger) -> torch.Tensor:
    """Read (item_a, item_b) positive pairs and map item_id -> dataset row.

    Accepts .parquet or .csv with columns item_a, item_b. Pairs with an unknown
    id are dropped (count is logged).
    """
    import pandas as pd
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    row = {iid: i for i, iid in enumerate(item_ids)}
    a = df["item_a"].astype(str).map(row)
    b = df["item_b"].astype(str).map(row)
    keep = a.notna() & b.notna()
    logger.info(f"  positive pairs: {int(keep.sum())} kept of {len(df)} (unknown ids dropped)")
    if int(keep.sum()) == 0:
        raise ValueError(f"no valid positive pairs in {path}")
    arr = np.stack([a[keep].to_numpy(), b[keep].to_numpy()], axis=1).astype(np.int64)
    return torch.from_numpy(arr)


# --------------------------------------------------------------------------- #
# Pure metric helpers (no torch; unit-testable)
# --------------------------------------------------------------------------- #
def usage_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p + 1e-12)).sum())


def utilisation_deficit(entropy: float, codebook_size: int) -> float:
    if codebook_size <= 1:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - entropy / math.log(codebook_size))))


def gini_coefficient(values) -> float:
    """Gini inequality index over a frequency distribution (0 = uniform, 1 = one bin)."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if n == 0:
        return 0.0
    total = x.sum()
    if total <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * x).sum()) / (n * total) - (n + 1) / n)


def effective_cells(n_items: int, n_unique: int) -> int:
    """Effective capacity via birthday inversion.

    Given N items hashing into M cells, observed U unique cells, solve
    U/N = (1 - exp(-x)) / x with x = N/M, then M = N/x. This recovers the
    *total* number of addressable cells from the collision count.
    """
    if n_items <= 0 or n_unique <= 0:
        return 0
    if n_unique >= n_items:
        return n_unique
    ratio = n_unique / n_items
    lo, hi = 1e-9, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        lhs = (1.0 - math.exp(-mid)) / mid if mid > 0 else 1.0
        if lhs > ratio:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    return int(round(n_items / x))


def tuple_redundancy(marginal_entropies: list[float], joint_entropy: float) -> tuple[float, float]:
    """Total correlation of the tuple (nats, bits)."""
    marginal_sum = sum(marginal_entropies)
    nats = max(0.0, marginal_sum - joint_entropy)
    return nats, nats * math.log2(math.e)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ResidualQuantizer(nn.Module):
    """Plain residual vector quantizer (random init, no k-means, no revival)."""

    def __init__(self, codebook_sizes: list[int], hidden_dim: int) -> None:
        super().__init__()
        self.codebook_sizes = list(codebook_sizes)
        self.num_stages = len(codebook_sizes)
        self.codebooks = nn.ParameterList()
        for k in codebook_sizes:
            cb = nn.Parameter(torch.empty(k, hidden_dim))
            nn.init.trunc_normal_(cb, std=0.02, a=-0.04, b=0.04)
            self.codebooks.append(cb)

    def forward(self, z: torch.Tensor, return_residuals: bool = False) -> dict:
        remainder = z
        quantized_sum = torch.zeros_like(z)
        indices_list: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = [z.detach()] if return_residuals else []
        cb_loss = z.new_zeros(())
        cm_loss = z.new_zeros(())
        for codebook in self.codebooks:
            e_sq = (codebook * codebook).sum(dim=-1)          # (K,)
            logits = 2.0 * remainder @ codebook.t() - e_sq    # (N, K)
            idx = logits.argmax(dim=-1)
            quantized = codebook[idx]                          # (N, D)
            cb_loss = cb_loss + sq_err(quantized, remainder.detach())
            cm_loss = cm_loss + sq_err(remainder, quantized.detach())
            indices_list.append(idx)
            quantized_sum = quantized_sum + quantized
            remainder = remainder - quantized.detach()
            if return_residuals:
                residuals.append(remainder.detach())
        indices = torch.stack(indices_list, dim=-1)            # (N, L)
        return {
            "quantized_sum": quantized_sum,
            "indices": indices,
            "residuals": residuals,
            "cb_loss": cb_loss,
            "cm_loss": cm_loss,
        }

    @torch.inference_mode()
    def encode_indices(self, z: torch.Tensor) -> torch.Tensor:
        remainder = z
        indices_list: list[torch.Tensor] = []
        for codebook in self.codebooks:
            e_sq = (codebook * codebook).sum(dim=-1)
            logits = 2.0 * remainder @ codebook.t() - e_sq
            idx = logits.argmax(dim=-1)
            indices_list.append(idx)
            remainder = remainder - codebook[idx]
        return torch.stack(indices_list, dim=-1)


class DeadCodeReviver:
    """Revives under-used codewords from a ring buffer of recent stage inputs.

    Per step: store each stage's input residual in the buffer and update an EMA of per-code
    usage. On revival: codes with EMA usage below threshold_frac * mean are overwritten by random
    buffered residuals, and their Adam moments are zeroed.
    """

    def __init__(self, codebook_sizes: list[int], hidden_dim: int, device: torch.device,
                 buffer_size: int = BUFFER_CAPACITY_PER_STAGE, decay: float = USAGE_DECAY,
                 threshold_frac: float = DEAD_CODE_THRESHOLD_FRAC) -> None:
        self.codebook_sizes = list(codebook_sizes)
        self.num_stages = len(codebook_sizes)
        self.buffer_size = buffer_size
        self.decay = decay
        self.threshold_frac = threshold_frac
        self.device = device
        self.buffers = [torch.zeros(buffer_size, hidden_dim, device=device) for _ in range(self.num_stages)]
        self.ptrs = [0] * self.num_stages
        self.has_wrapped = [False] * self.num_stages
        self.usage: list[torch.Tensor | None] = [None] * self.num_stages

    @torch.no_grad()
    def update(self, residuals: list[torch.Tensor], indices: torch.Tensor) -> None:
        """residuals[d] = input to stage d; indices is (N, L)."""
        for d in range(self.num_stages):
            res = residuals[d]
            n = min(res.shape[0], self.buffer_size)
            res = res[:n].to(device=self.device, dtype=self.buffers[d].dtype)
            ptr = self.ptrs[d]
            first = min(n, self.buffer_size - ptr)
            self.buffers[d][ptr:ptr + first] = res[:first]
            if n > first:
                self.buffers[d][:n - first] = res[first:]
            if ptr + n >= self.buffer_size:
                self.has_wrapped[d] = True
            self.ptrs[d] = (ptr + n) % self.buffer_size

            counts = torch.bincount(indices[:, d], minlength=self.codebook_sizes[d]).float()
            if self.usage[d] is None:
                self.usage[d] = counts.clone()
            else:
                self.usage[d].mul_(self.decay).add_(counts, alpha=1.0 - self.decay)

    @torch.no_grad()
    def revive(self, codebooks, optimizer: torch.optim.Optimizer) -> int:
        total = 0
        for d, cb_param in enumerate(codebooks):
            usage = self.usage[d]
            if usage is None:
                continue
            valid = self.buffer_size if self.has_wrapped[d] else self.ptrs[d]
            if valid == 0:
                continue
            total_usage = float(usage.sum().item())
            k = cb_param.shape[0]
            # Relative cutoff: an absolute count would rescale with batch and codebook size.
            cutoff = self.threshold_frac * (total_usage / k)
            dead = (usage < cutoff).nonzero(as_tuple=True)[0]
            if dead.numel() == 0:
                continue
            rand = torch.randint(0, valid, (dead.numel(),), device=self.device)
            cb_param.data[dead] = self.buffers[d][rand].to(dtype=cb_param.dtype)
            usage[dead] = max(1.0, total_usage / k)
            state = optimizer.state.get(cb_param, {})
            if "exp_avg" in state:
                state["exp_avg"][dead] = 0.0
            if "exp_avg_sq" in state:
                state["exp_avg_sq"][dead] = 0.0
            total += int(dead.numel())
        return total


class ContinuousVAE(nn.Module):
    """Encoder/decoder only — the geometry shared by both arms (no codebook)."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class RQVAE(nn.Module):
    """Baseline: geometry + codebook trained jointly through STE."""

    def __init__(self, input_dim: int, hidden_dim: int, codebook_sizes: list[int],
                 beta: float) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim)
        self.quantizer = ResidualQuantizer(codebook_sizes, hidden_dim)
        self.beta = beta
        self.codebook_sizes = list(codebook_sizes)

    def forward(self, x: torch.Tensor, return_residuals: bool = False) -> dict:
        z = self.encoder(x)
        q = self.quantizer(z, return_residuals=return_residuals)
        # Straight-through estimator: decoder gradient flows to z as-is.
        z_ste = z + (q["quantized_sum"] - z).detach()
        x_hat = self.decoder(z_ste)
        recon_loss = sq_err(x_hat, x)
        loss = recon_loss + q["cb_loss"] + self.beta * q["cm_loss"]
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "cb_loss": q["cb_loss"],
            "cm_loss": q["cm_loss"],
            "indices": q["indices"],
            "quantized_sum": q["quantized_sum"],
            "residuals": q["residuals"],
        }

    @torch.inference_mode()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer.encode_indices(self.encoder(x))


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def encode_latents(vae: ContinuousVAE, embeddings: torch.Tensor,
                   batch_size: int, device: torch.device) -> torch.Tensor:
    """Encode the full corpus to latent vectors (returned on CPU)."""
    vae.eval()
    zs: list[torch.Tensor] = []
    for i in range(0, len(embeddings), batch_size):
        x = embeddings[i:i + batch_size].to(device)
        zs.append(vae.encode(x).cpu())
    return torch.cat(zs, dim=0)


@torch.inference_mode()
def encode_indices_full(model: RQVAE, embeddings: torch.Tensor,
                        batch_size: int, device: torch.device) -> torch.Tensor:
    """Encode the full corpus to SID tuples (returned on CPU)."""
    model.eval()
    idxs: list[torch.Tensor] = []
    for i in range(0, len(embeddings), batch_size):
        x = embeddings[i:i + batch_size].to(device)
        idxs.append(model.encode_indices(x).cpu())
    return torch.cat(idxs, dim=0)


def assign_nearest(x: torch.Tensor, centroids: torch.Tensor,
                   batch_size: int) -> torch.Tensor:
    """Nearest-centroid labels for every row of x (device-agnostic)."""
    n = x.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=x.device)
    c_sq = (centroids * centroids).sum(dim=-1)                # (K,)
    for s in range(0, n, batch_size):
        xb = x[s:s + batch_size]
        d2 = (xb * xb).sum(dim=-1, keepdim=True) - 2.0 * xb @ centroids.t() + c_sq
        labels[s:s + batch_size] = d2.argmin(dim=-1)
    return labels


def fit_kmeans(x_np: np.ndarray, k: int, *, niter: int, nredo: int, seed: int,
               backend: str, prefer_gpu: bool,
               logger: logging.Logger) -> tuple[np.ndarray, str]:
    """Fit k-means on x_np; returns (centroids (k, d) float32, backend name).

    Prefers FAISS (GPU when available), falls back to FAISS-CPU, then sklearn.
    """
    x = np.ascontiguousarray(x_np, dtype=np.float32)
    if backend in ("auto", "faiss"):
        try:
            import faiss
        except ImportError:
            if backend == "faiss":
                raise
            logger.info("    faiss not installed; falling back to sklearn KMeans")
        else:
            use_gpu = prefer_gpu
            if use_gpu:
                try:
                    use_gpu = faiss.get_num_gpus() > 0
                except Exception:
                    use_gpu = False
            for gpu_flag in ([True, False] if use_gpu else [False]):
                try:
                    km = faiss.Kmeans(
                        x.shape[1], k, niter=niter, nredo=nredo, verbose=False,
                        min_points_per_centroid=1, seed=seed, gpu=gpu_flag,
                    )
                    km.train(x)
                    name = "faiss-gpu" if gpu_flag else "faiss-cpu"
                    return km.centroids.astype(np.float32), name
                except Exception as exc:  # noqa: BLE001
                    logger.info(f"    faiss kmeans (gpu={gpu_flag}) failed: {exc}")
    km = KMeans(n_clusters=k, n_init=nredo, random_state=seed,
                max_iter=niter, tol=1e-4).fit(x)
    return km.cluster_centers_.astype(np.float32), "sklearn"


def hierarchical_kmeans(z_all: torch.Tensor, codebook_sizes: list[int],
                        logger: logging.Logger, fit_samples: int, n_init: int,
                        niter: int, backend: str, prefer_gpu: bool,
                        random_state: int, assign_batch_size: int,
                        fit_pool: torch.Tensor | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
    """DRQ stage 2: residual k-means on frozen latents.

    Fits on a subsample (fast), assigns the whole corpus with batched torch
    matmul, recurses on residuals. `fit_pool` limits which rows may be used to
    *fit* (e.g. train-only); assignment always covers the whole corpus.
    """
    n = z_all.shape[0]
    pool = z_all if fit_pool is None else fit_pool
    n_pool = pool.shape[0]
    fit_n = min(int(fit_samples), n_pool)
    rng = np.random.default_rng(random_state)
    fit_idx = (rng.choice(n_pool, size=fit_n, replace=False)
               if fit_n < n_pool else np.arange(n_pool))

    residual_all = z_all
    residual_pool = pool
    centroids: list[torch.Tensor] = []
    idxs: list[torch.Tensor] = []
    for stage, k in enumerate(codebook_sizes):
        k_eff = min(int(k), fit_n)
        c_np, backend_used = fit_kmeans(
            residual_pool[fit_idx].numpy(), k_eff, niter=niter, nredo=n_init,
            seed=random_state, backend=backend, prefer_gpu=prefer_gpu,
            logger=logger,
        )
        c = torch.from_numpy(c_np)
        centroids.append(c)
        labels = assign_nearest(residual_all, c, assign_batch_size)
        idxs.append(labels)
        residual_all = residual_all - c[labels]
        residual_pool = residual_pool - c[assign_nearest(residual_pool, c, assign_batch_size)]
        mean_sq = float((residual_all * residual_all).sum(dim=-1).mean().item())
        logger.info(f"    level {stage}: k-means K={k_eff} via {backend_used}, "
                    f"residual ||r||^2 mean = {mean_sq:.4f}")
    return centroids, torch.stack(idxs, dim=-1)


# --------------------------------------------------------------------------- #
# Diagnostics (DRQ injectivity metrics)
# --------------------------------------------------------------------------- #
def auto_sigma(centroids: torch.Tensor, counts: torch.Tensor | None = None,
               max_codes: int = 2048) -> float:
    """Per-level kernel width, calibrated to the centroid spacing.

    sigma = SIGMA_TOL * median nearest-neighbour distance among *used* codes.
    With SIGMA_TOL = 0.1, two codes closer than ~10% of the typical spacing are
    treated as indistinguishable (weight ~1), and codes further apart than one
    typical spacing contribute ~0. So O_pi stays within a few percent of the
    distribution floor 1/sum(pi^2) whenever the codebook is genuinely spread,
    and blows up to ~1 only when codes genuinely collide.
    """
    c = centroids.float()
    if counts is not None and int((counts > 0).sum()) >= 2:
        c = c[counts > 0]
    if c.shape[0] < 2:
        return 1.0
    if c.shape[0] > max_codes:
        idx = torch.linspace(0, c.shape[0] - 1, max_codes).long()
        c = c[idx]
    d = torch.cdist(c, c, p=2.0)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    nn = nn[torch.isfinite(nn)]
    if nn.numel() == 0:
        return 1.0
    s = float(nn.median().item())
    # s == 0 means exact duplicate centroids exist; a tiny sigma keeps their
    # weight at 1 while suppressing every genuinely separated pair.
    return SIGMA_TOL * s if s > 0 else 1e-6


def _overlap_metrics(counts: torch.Tensor, centroids: torch.Tensor,
                     sigma: float) -> dict:
    """O_pi = sum_ij pi_i pi_j exp(-||c_i-c_j||^2 / 4 sigma^2), and friends.

    Decomposition (DRQ, arXiv:2606.01844):
        O_pi = sum_i pi_i^2                       <- distribution floor
             + sum_{i != j} pi_i pi_j exp(...)    <- geometry cross-overlap
        K_eff = 1 / O_pi
    """
    total = float(counts.sum().item())
    if total <= 0:
        return {"overlap_rate": 0.0, "effective_codes": 0.0,
                "distribution_floor": 0.0, "geometry_cross_overlap": 0.0,
                "participation_ratio": 0.0, "sigma": sigma}
    pi = counts.float() / total
    distribution_floor = float((pi * pi).sum().item())
    sigma_used = sigma if sigma > 0 else auto_sigma(centroids, counts)
    c = centroids.float()
    d2 = torch.cdist(c, c, p=2.0).pow(2.0)
    w = torch.exp(-d2 / (4.0 * sigma_used * sigma_used))
    o_pi = float((pi @ w @ pi).item())
    return {
        "overlap_rate": o_pi,
        "effective_codes": (1.0 / o_pi) if o_pi > 0 else 0.0,
        "distribution_floor": distribution_floor,
        "geometry_cross_overlap": max(0.0, o_pi - distribution_floor),
        # sigma -> 0 limit of K_eff: ignores geometry, pure usage imbalance.
        "participation_ratio": (1.0 / distribution_floor) if distribution_floor > 0 else 0.0,
        "sigma": sigma_used,
    }


def sigma_scan(counts: torch.Tensor, centroids: torch.Tensor, base_sigma: float,
               multipliers: list[float]) -> list[dict]:
    """O_pi / K_eff at several kernel widths around the auto sigma.

    Sigma is metric-only (never touches training), so sweeping it is free.
    Each level is calibrated by ITS OWN centroid spacing, so reporting
    K_eff(base_sigma * m) makes the two arms comparable as *curves* rather
    than at one arbitrary sigma: a spread codebook holds K_eff near the
    participation ratio across the whole scan, a colliding one collapses
    already at m < 1.
    """
    return [
        {"sigma_mult": m, **_overlap_metrics(counts, centroids, base_sigma * m)}
        for m in multipliers
    ]


def compute_stage_diagnostics(indices: torch.Tensor, codebooks: list[torch.Tensor],
                              sigma: float, scan_mults: list[float] | None = None,
                              ) -> list[dict]:
    """Per-level injectivity diagnostics."""
    n, l = indices.shape
    out: list[dict] = []
    for stage in range(l):
        idx = indices[:, stage]
        k = codebooks[stage].shape[0]
        counts = torch.bincount(idx, minlength=k)
        used = int((counts > 0).sum().item())
        dead_pct = (1.0 - used / k) * 100.0
        max_freq = float(counts.max().item()) / n
        p = counts[counts > 0].float() / n
        entropy = float(-(p * torch.log(p + 1e-12)).sum().item())
        perplexity = math.exp(entropy)
        gini = gini_coefficient(counts.numpy())
        om = _overlap_metrics(counts, codebooks[stage], sigma)
        out.append({
            "stage": stage,
            "total_codes": k,
            "used_codes": used,
            "dead_code_pct": dead_pct,
            "max_freq": max_freq,
            "entropy": entropy,
            "perplexity": perplexity,
            "gini": gini,
            "utilisation_deficit": utilisation_deficit(entropy, k),
            "distribution_floor": om["distribution_floor"],
            "participation_ratio": om["participation_ratio"],
            "overlap_rate": om["overlap_rate"],
            "effective_codes": om["effective_codes"],
            "geometry_cross_overlap": om["geometry_cross_overlap"],
            "kernel_sigma": om["sigma"],
        })
        if scan_mults:
            out[stage]["sigma_scan"] = sigma_scan(counts, codebooks[stage],
                                                  om["sigma"], scan_mults)
    return out


def compute_tuple_diagnostics(indices: torch.Tensor) -> dict:
    """Full-tuple collision and capacity diagnostics."""
    n, l = indices.shape
    unique, counts = torch.unique(indices, dim=0, return_counts=True)
    n_unique = int(unique.shape[0])
    counts_i = counts.int()
    max_bucket = int(counts_i.max().item()) if n_unique else 0
    n_colliding_tuples = int((counts_i > 1).sum().item())
    singleton_items = int((counts_i == 1).sum().item())
    items_in_collision = n - singleton_items
    # DRQ's definition (arXiv:2606.01844): collision_rate = 1 - unique/N.
    # (Sanity check on the paper's headline numbers: 1 - 385,551/500,000 =
    #  0.228898 and 1 - 462,660/500,000 = 0.074680, i.e. N = 500k there.)
    collision_rate = 1.0 - (n_unique / n) if n else 0.0
    items_in_collision_frac = items_in_collision / n if n else 0.0
    p = counts.float() / n
    joint_entropy = float(-(p * torch.log(p + 1e-12)).sum().item())
    joint_perplexity = math.exp(joint_entropy)
    return {
        "n_items": n,
        "tuple_length": l,
        "n_unique_tuples": n_unique,
        "uniqueness_ratio": (n_unique / n) if n else 0.0,
        "collision_rate": collision_rate,
        "items_in_collision_frac": items_in_collision_frac,
        "items_in_collision": items_in_collision,
        "n_colliding_tuples": n_colliding_tuples,
        "max_collision_bucket": max_bucket,
        "mean_collision_bucket": (n / n_unique) if n_unique else 0.0,
        "joint_entropy": joint_entropy,
        "joint_perplexity": joint_perplexity,
        "effective_cells": effective_cells(n, n_unique),
    }


def compute_prefix_uniqueness(indices: torch.Tensor) -> list[float]:
    """Fraction of unique prefixes at each depth (1..L)."""
    n, l = indices.shape
    ratios: list[float] = []
    for depth in range(1, l + 1):
        sub = indices[:, :depth]
        ratios.append(torch.unique(sub, dim=0).shape[0] / n)
    return ratios


def run_diagnostics(indices: torch.Tensor, codebooks: list[torch.Tensor],
                    sigma: float, scan_mults: list[float] | None = None,
                    ) -> dict:
    """Full injectivity report for one arm."""
    stage = compute_stage_diagnostics(indices, codebooks, sigma, scan_mults)
    tuples = compute_tuple_diagnostics(indices)
    marginal_entropies = [m["entropy"] for m in stage]
    red_nats, red_bits = tuple_redundancy(marginal_entropies, tuples["joint_entropy"])
    return {
        "stage": stage,
        "tuples": tuples,
        "prefix_uniqueness": compute_prefix_uniqueness(indices),
        "redundancy_nats": red_nats,
        "redundancy_bits": red_bits,
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _make_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int,
                           min_lr_ratio: float) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


@torch.inference_mode()
def evaluate_continuous_recon(model, loader: DataLoader, device: torch.device) -> float:
    """MSE of decoder(encoder(x)) — ignores quantization entirely.

    Identical computation for both arms, so it is the fair *geometry* number,
    but it says nothing about what the discrete code actually costs you.
    """
    model.eval()
    total, count = 0.0, 0
    for x in loader:
        x = x.to(device)
        x_hat = model.decoder(model.encoder(x))
        total += sq_err(x_hat, x).item() * x.shape[0]
        count += x.shape[0]
    return total / max(1, count)


@torch.inference_mode()
def evaluate_quantized_recon_rqvae(model: "RQVAE", loader: DataLoader,
                                   device: torch.device) -> float:
    """MSE of decoder(quantized latent) — includes quantization error."""
    model.eval()
    total, count = 0.0, 0
    for x in loader:
        x = x.to(device)
        q = model.quantizer(model.encoder(x))
        total += sq_err(model.decoder(q["quantized_sum"]), x).item() * x.shape[0]
        count += x.shape[0]
    return total / max(1, count)


@torch.inference_mode()
def evaluate_quantized_recon_drq(vae: "ContinuousVAE",
                                 centroids: list[torch.Tensor],
                                 loader: DataLoader,
                                 device: torch.device) -> float:
    """MSE of decoder(sum of assigned centroids) — the deployed reconstruction."""
    vae.eval()
    cb = [c.to(device) for c in centroids]
    total, count = 0.0, 0
    for x in loader:
        x = x.to(device)
        remainder = vae.encoder(x)
        zq = torch.zeros_like(remainder)
        for c in cb:
            c_sq = (c * c).sum(dim=-1)
            idx = (2.0 * remainder @ c.t() - c_sq).argmax(dim=-1)
            zq = zq + c[idx]
            remainder = remainder - c[idx]
        total += sq_err(vae.decoder(zq), x).item() * x.shape[0]
        count += x.shape[0]
    return total / max(1, count)


def _batch_stage_perplexity(indices: torch.Tensor, k: int, stage: int) -> float:
    counts = torch.bincount(indices[:, stage], minlength=k).float()
    p = counts[counts > 0]
    if p.numel() == 0:
        return 0.0
    p = p / p.sum()
    return math.exp(float(-(p * torch.log(p + 1e-12)).sum().item()))


def train_continuous_vae(vae: ContinuousVAE, train_loader: DataLoader,
                         val_loader: DataLoader, epochs: int, lr: float,
                         warmup_steps: int, min_lr_ratio: float,
                         device: torch.device, logger: logging.Logger,
                         embeddings: torch.Tensor | None = None,
                         pairs: torch.Tensor | None = None,
                         lam: float = 0.0, tau: float = 0.1,
                         pair_batch: int = 1024) -> ContinuousVAE:
    """DRQ stage 1: L = E||x - D(E(x))||^2 + lam * L_contrastive (paper Eq. 2).

    lam = 0 is the paper's DRQ-VAE; lam = 0.3 is DRQ-VAE+CL.
    The contrastive term uses positive pairs (row indices into `embeddings`).
    """
    if lam > 0 and (pairs is None or embeddings is None):
        raise ValueError("lam > 0 needs positive pairs (--drq_pairs)")
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = _make_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio)
    step = 0
    for epoch in range(epochs):
        vae.train()
        running = defaultdict(float)
        t0 = time.time()
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            z, x_hat = vae(x)
            recon = sq_err(x_hat, x)
            loss = recon
            if lam > 0:
                pi = torch.randint(0, pairs.shape[0], (pair_batch,))
                xa = embeddings[pairs[pi, 0]].to(device)
                xb = embeddings[pairs[pi, 1]].to(device)
                cl = info_nce(vae.encoder(xa), vae.encoder(xb), tau)
                loss = recon + lam * cl
                running["cl"] += cl.item()
            running["recon"] += recon.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1
            running["loss"] += loss.item()
            if step % 100 == 0:
                logger.info(
                    f"    [DRQ stage1] epoch {epoch + 1}/{epochs} step {step} "
                    f"loss={loss.item():.4f} recon={recon.item():.4f} "
                    f"cl={running['cl'] / max(1, step - (epoch * steps_per_epoch)):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
        val_recon = evaluate_continuous_recon(vae, val_loader, device)
        logger.info(
            f"  [DRQ stage1] epoch {epoch + 1}/{epochs} "
            f"train_loss={running['loss'] / max(1, steps_per_epoch):.4f} "
            f"train_recon={running['recon'] / max(1, steps_per_epoch):.4f} "
            f"val_recon={val_recon:.4f} ({time.time() - t0:.1f}s)"
        )
    return vae


def _joint_warmup_steps(warmup_steps: int, ae_warmup_epochs: int) -> int:
    """LR ramp for the joint phase. After an AE warmup the cosine restarts with a
    shorter ramp: the quantizer enters a converged AE and its initial STE
    gradients are large, so a 2000-step ramp would waste most of the schedule."""
    if ae_warmup_epochs > 0:
        return min(warmup_steps, HANDOFF_WARMUP_STEPS)
    return warmup_steps


def train_rqvae(model: RQVAE, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float, warmup_steps: int, min_lr_ratio: float,
                device: torch.device, logger: logging.Logger,
                dead_code_fix: bool = DEAD_CODE_FIX,
                dead_code_interval: int = DEAD_CODE_FIX_INTERVAL,
                dead_code_threshold_frac: float = DEAD_CODE_THRESHOLD_FRAC,
                usage_decay: float = USAGE_DECAY,
                ae_warmup_epochs: int = AE_WARMUP_EPOCHS) -> RQVAE:
    """Baseline: joint geometry + codebook training through STE, optional dead-code revival.

    Optional AE warmup (ae_warmup_epochs > 0) runs reconstruction-only epochs on encoder+decoder
    with the same optimizer first; the LR schedule starts after it, as in the reference script.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    if ae_warmup_epochs > 0:
        logger.info(f"  AE warmup ENABLED: {ae_warmup_epochs} reconstruction-only epoch(s)")
        for ep in range(ae_warmup_epochs):
            model.train()
            total = 0.0
            for x in train_loader:
                x = x.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = sq_err(model.decoder(model.encoder(x)), x)
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.info(f"  [AE warmup] epoch {ep + 1}/{ae_warmup_epochs} "
                        f"recon={total / max(1, len(train_loader)):.6f}")
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    # Cosine restart at the handoff: after AE warmup the quantizer is being
    # introduced into a converged AE, so give it its own (shorter) ramp instead
    # of the full global warmup.
    eff_warmup = _joint_warmup_steps(warmup_steps, ae_warmup_epochs)
    if ae_warmup_epochs > 0 and eff_warmup != warmup_steps:
        logger.info(f"  schedule restarts at handoff: warmup {warmup_steps} -> "
                    f"{eff_warmup} steps, cosine over {total_steps} steps")
    scheduler = _make_cosine_scheduler(optimizer, eff_warmup, total_steps, min_lr_ratio)
    k0, k1 = model.codebook_sizes[0], model.codebook_sizes[-1]
    reviver = DeadCodeReviver(
        codebook_sizes=model.codebook_sizes,
        hidden_dim=model.quantizer.codebooks[0].shape[1],
        device=device,
        decay=usage_decay,
        threshold_frac=dead_code_threshold_frac,
    ) if dead_code_fix else None
    if reviver is not None:
        logger.info(f"  dead-code revival ENABLED: every {dead_code_interval} steps, "
                    f"< {dead_code_threshold_frac:.1%} of mean usage, until "
                    f"{DEAD_CODE_REVIVE_UNTIL_FRAC:.0%} of steps")
    else:
        logger.info("  dead-code revival DISABLED")
    revived_total = 0
    step = 0
    for epoch in range(epochs):
        model.train()
        running = defaultdict(float)
        t0 = time.time()
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x, return_residuals=reviver is not None)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1
            if reviver is not None:
                reviver.update(out["residuals"], out["indices"])
                if step % dead_code_interval == 0 and step < DEAD_CODE_REVIVE_UNTIL_FRAC * total_steps:
                    n_rev = reviver.revive(model.quantizer.codebooks, optimizer)
                    revived_total += n_rev
                    if n_rev:
                        logger.info(f"    [dead-code] step {step}: revived {n_rev} code(s)")
            running["loss"] += out["loss"].item()
            running["recon"] += out["recon_loss"].item()
            running["cb"] += out["cb_loss"].item()
            running["cm"] += out["cm_loss"].item()
            if step % 100 == 0:
                p0 = _batch_stage_perplexity(out["indices"], k0, 0)
                pl = _batch_stage_perplexity(out["indices"], k1, len(model.codebook_sizes) - 1)
                logger.info(
                    f"    [RQVAE] epoch {epoch + 1}/{epochs} step {step} "
                    f"loss={out['loss'].item():.4f} cb={out['cb_loss'].item():.4f} "
                    f"cm={out['cm_loss'].item():.4f} perp0={p0:.1f} perpL={pl:.1f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
        val_recon = evaluate_continuous_recon(model, val_loader, device)
        n = max(1, steps_per_epoch)
        logger.info(
            f"  [RQVAE] epoch {epoch + 1}/{epochs} "
            f"loss={running['loss'] / n:.4f} recon={running['recon'] / n:.4f} "
            f"cb={running['cb'] / n:.4f} cm={running['cm'] / n:.4f} "
            f"val_recon={val_recon:.4f} revived_total={revived_total} ({time.time() - t0:.1f}s)"
        )
    return model


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def run_rqvae_arm(dataset: EmbeddingDataset, train_idx: np.ndarray, val_idx: np.ndarray,
                  codebook_sizes: list[int], args, logger: logging.Logger,
                  name: str = "rqvae",
                  ae_warmup_epochs: int | None = None) -> dict:
    warmup_epochs = args.ae_warmup_epochs if ae_warmup_epochs is None else ae_warmup_epochs
    if name == "warmup":
        logger.info(f"=== Arm 3: AE warmup ({warmup_epochs} recon-only epochs) then joint RQ-VAE ===")
    else:
        logger.info("=== Arm 1: baseline RQ-VAE (joint training) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RQVAE(args.input_dim, args.hidden_dim, codebook_sizes, args.beta).to(device)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    train_rqvae(model, train_loader, val_loader, args.epochs, args.lr,
                args.warmup_steps, args.min_lr_ratio, device, logger,
                dead_code_fix=args.dead_code_fix,
                ae_warmup_epochs=warmup_epochs,
                dead_code_interval=args.dead_code_fix_interval,
                dead_code_threshold_frac=args.dead_code_threshold_frac,
                usage_decay=args.usage_decay)
    indices = encode_indices_full(model, dataset.embeddings, args.inference_batch_size, device)
    codebooks = [cb.detach().cpu() for cb in model.quantizer.codebooks]
    return {
        "indices": indices,
        "codebooks": codebooks,
        "model": model,
        "recon_continuous": evaluate_continuous_recon(model, val_loader, device),
        "recon_quantized": evaluate_quantized_recon_rqvae(model, val_loader, device),
    }


def run_drq_arm(dataset: EmbeddingDataset, train_idx: np.ndarray, val_idx: np.ndarray,
                codebook_sizes: list[int], args, logger: logging.Logger) -> dict:
    logger.info("=== Arm 2: DRQ (separate training: VAE then k-means) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = ContinuousVAE(args.input_dim, args.hidden_dim).to(device)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    # Stage 1: continuous VAE (recon only)
    pairs = None
    if args.drq_lambda > 0:
        pairs = load_positive_pairs(args.drq_pairs, dataset.item_ids, logger) if args.drq_pairs             else None
        if pairs is None:
            raise ValueError("--drq_lambda > 0 requires --drq_pairs")
    train_continuous_vae(vae, train_loader, val_loader, args.epochs, args.lr,
                         args.warmup_steps, args.min_lr_ratio, device, logger,
                         embeddings=dataset.embeddings, pairs=pairs,
                         lam=args.drq_lambda, tau=args.drq_tau,
                         pair_batch=args.drq_pair_batch)
    # Stage 2: hierarchical k-means on frozen full-corpus latents
    z_all = encode_latents(vae, dataset.embeddings, args.inference_batch_size, device)
    fit_pool = z_all if args.kmeans_fit_on == "all" else z_all[torch.as_tensor(train_idx)]
    logger.info(f"  k-means fit pool: {fit_pool.shape[0]} of {z_all.shape[0]} latents "
                f"(--kmeans_fit_on {args.kmeans_fit_on})")
    centroids, indices = hierarchical_kmeans(
        z_all, codebook_sizes, logger, args.kmeans_fit_samples,
        args.kmeans_n_init, args.kmeans_niter, args.kmeans_backend,
        torch.cuda.is_available(), args.seed, args.inference_batch_size,
        fit_pool=fit_pool,
    )
    return {
        "indices": indices,
        "codebooks": centroids,
        "model": vae,
        "recon_continuous": evaluate_continuous_recon(vae, val_loader, device),
        "recon_quantized": evaluate_quantized_recon_drq(vae, centroids, val_loader, device),
    }


# --------------------------------------------------------------------------- #
# Output / comparison
# --------------------------------------------------------------------------- #
def build_index(dataset: EmbeddingDataset, indices: torch.Tensor) -> dict:
    return {
        item_id: [int(v) for v in row.tolist()]
        for item_id, row in zip(dataset.item_ids, indices)
    }


def build_comparison(baseline_diag: dict, candidate_diag: dict) -> dict:
    def stage_headline(diag: dict) -> dict:
        s0 = diag["stage"][0]
        return {
            "level0_perplexity": s0["perplexity"],
            "level0_gini": s0["gini"],
            "level0_effective_codes": s0["effective_codes"],
            "level0_overlap_rate": s0["overlap_rate"],
        }

    return {
        "baseline": {
            "tuples": baseline_diag["tuples"],
            "stage_headline": stage_headline(baseline_diag),
        },
        "candidate": {
            "tuples": candidate_diag["tuples"],
            "stage_headline": stage_headline(candidate_diag),
        },
    }


def _sigma_scan_table(baseline_diag: dict, candidate_diag: dict,
                      logger: logging.Logger) -> None:
    """Per-level K_eff(O_pi) across the sigma scan for both sides."""
    for stage_a, stage_b in zip(baseline_diag["stage"], candidate_diag["stage"]):
        scan_a, scan_b = stage_a.get("sigma_scan"), stage_b.get("sigma_scan")
        if not scan_a or not scan_b:
            continue
        logger.info(f"--- level {stage_a['stage']} sigma scan "
                    f"(baseline auto sigma = {stage_a['kernel_sigma']:.4g}, "
                    f"candidate = {stage_b['kernel_sigma']:.4g}) ---")
        rows = []
        for a, b in zip(scan_a, scan_b):
            rows.append((f"x{a['sigma_mult']:g}",
                         f"{a['sigma']:.4g}", f"{a['effective_codes']:.1f}",
                         f"{b['sigma']:.4g}", f"{b['effective_codes']:.1f}"))
        log_table(logger, rows, ("mult", "base sigma", "base K_eff",
                                 "cand sigma", "cand K_eff"))


def print_comparison(comparison: dict, logger: logging.Logger, title: str = "",
                     base_name: str = "rqvae", cand_name: str = "drq") -> None:
    r = comparison["baseline"]
    d = comparison["candidate"]
    rt, dt = r["tuples"], d["tuples"]
    logger.info("")
    logger.info("=" * 78)
    logger.info(f"COMPARISON  ({base_name} baseline  vs  {cand_name})  {title}")
    logger.info("=" * 78)
    rows = [
        ("unique tuples", f"{rt['n_unique_tuples']:,}", f"{dt['n_unique_tuples']:,}"),
        ("uniqueness ratio", f"{rt['uniqueness_ratio']:.4f}", f"{dt['uniqueness_ratio']:.4f}"),
        ("collision rate", f"{rt['collision_rate']:.4f}", f"{dt['collision_rate']:.4f}"),
        ("max collision bucket", f"{rt['max_collision_bucket']}", f"{dt['max_collision_bucket']}"),
        ("effective cells", f"{rt['effective_cells']:,}", f"{dt['effective_cells']:,}"),
        ("joint perplexity", f"{rt['joint_perplexity']:.1f}", f"{dt['joint_perplexity']:.1f}"),
        ("level0 perplexity", f"{r['stage_headline']['level0_perplexity']:.1f}",
         f"{d['stage_headline']['level0_perplexity']:.1f}"),
        ("level0 gini", f"{r['stage_headline']['level0_gini']:.3f}",
         f"{d['stage_headline']['level0_gini']:.3f}"),
        ("level0 K_eff", f"{r['stage_headline']['level0_effective_codes']:.1f}",
         f"{d['stage_headline']['level0_effective_codes']:.1f}"),
        ("level0 O_pi", f"{r['stage_headline']['level0_overlap_rate']:.5f}",
         f"{d['stage_headline']['level0_overlap_rate']:.5f}"),
    ]
    log_table(logger, [(a, b, c) for a, b, c in rows],
              ("metric", base_name, cand_name))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DRQ vs RQ-VAE injectivity ablation")
    p.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH,
                   help="parquet or h5 file of precomputed item embeddings")
    p.add_argument("--arm", type=str,
                   choices=["rqvae", "drq", "warmup", "both", "all"], default="both",
                   help="rqvae|drq|warmup; both = rqvae+drq; all = rqvae+warmup+drq")
    p.add_argument("--output_dir", type=str, default="drq_vs_rqvae_out")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--input_dim", type=int, default=INPUT_DIM)
    p.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--codebook_size", type=int, default=CODEBOOK_SIZE)
    p.add_argument("--num_codebooks", type=int, default=NUM_CODEBOOKS)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--inference_batch_size", type=int, default=INFERENCE_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--min_lr_ratio", type=float, default=MIN_LR_RATIO)
    p.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    p.add_argument("--overlap_sigma", type=float, default=OVERLAP_SIGMA,
                   help="O_pi kernel width; 0 = auto (0.1 x median nearest-neighbour "
                        "centroid distance per level)")
    p.add_argument("--overlap_sigma_scan", type=float, nargs="*", default=SIGMA_SCAN_MULTS,
                   help="sigma multipliers for the per-level K_eff(O_pi) scan "
                        "(relative to the auto sigma of each level); empty = off")
    p.add_argument("--kmeans_fit_samples", type=int, default=KMEANS_FIT_SAMPLES,
                   help="rows subsampled to fit each k-means level (assignment uses all rows)")
    p.add_argument("--drq_lambda", type=float, default=DRQ_LAMBDA,
                   help="stage-1 contrastive weight; 0 = DRQ-VAE, 0.3 = DRQ-VAE+CL")
    p.add_argument("--drq_pairs", type=str, default=None,
                   help=".parquet/.csv with columns item_a,item_b (positive pairs)")
    p.add_argument("--drq_tau", type=float, default=DRQ_TAU)
    p.add_argument("--drq_pair_batch", type=int, default=DRQ_PAIR_BATCH)
    p.add_argument("--kmeans_n_init", type=int, default=KMEANS_N_INIT,
                   help="k-means restarts (nredo for faiss, n_init for sklearn)")
    p.add_argument("--kmeans_niter", type=int, default=KMEANS_NITER)
    p.add_argument("--kmeans_backend", type=str, choices=["auto", "faiss", "sklearn"],
                   default="auto")
    p.add_argument("--kmeans_fit_on", type=str, choices=["all", "train"], default="all",
                   help="'all' matches the DRQ paper (dataset-level assignment); "
                        "'train' restricts fitting to the train split")
    p.add_argument("--ae_warmup_epochs", type=int, default=AE_WARMUP_EPOCHS,
                   help="RQ-VAE arm: reconstruction-only epochs before joint training (0 = off)")
    p.add_argument("--warmup_arm_epochs", type=int, default=WARMUP_ARM_EPOCHS,
                   help="warmup arm: reconstruction-only epochs before the joint phase")
    p.add_argument("--dead_code_fix", action=argparse.BooleanOptionalAction, default=DEAD_CODE_FIX,
                   help="RQ-VAE arm: revive under-used codes from a buffer of recent residuals")
    p.add_argument("--dead_code_fix_interval", type=int, default=DEAD_CODE_FIX_INTERVAL)
    p.add_argument("--dead_code_threshold_frac", type=float, default=DEAD_CODE_THRESHOLD_FRAC)
    p.add_argument("--usage_decay", type=float, default=USAGE_DECAY)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--save_index", action="store_true")
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    run_tag = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, run_tag)
    logger = setup_logger(out_dir)
    logger.info(f"output dir: {out_dir}")

    codebook_sizes = [args.codebook_size] * args.num_codebooks

    dataset = EmbeddingDataset(args.input_path, args.max_samples, logger)
    n = len(dataset)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    val_n = max(1, int(n * args.val_ratio))
    train_idx = perm[val_n:]
    val_idx = perm[:val_n]
    logger.info(f"split: train={len(train_idx)} val={len(val_idx)}")

    if args.arm == "both":
        arms = ["rqvae", "drq"]
    elif args.arm == "all":
        arms = ["rqvae", "warmup", "drq"]
    else:
        arms = [args.arm]
    results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    for arm in arms:
        if arm == "rqvae":
            res = run_rqvae_arm(dataset, train_idx, val_idx, codebook_sizes, args, logger)
        elif arm == "warmup":
            res = run_rqvae_arm(dataset, train_idx, val_idx, codebook_sizes, args, logger,
                                name="warmup", ae_warmup_epochs=args.warmup_arm_epochs)
        else:
            res = run_drq_arm(dataset, train_idx, val_idx, codebook_sizes, args, logger)
        results[arm] = res

        diag = run_diagnostics(res["indices"], res["codebooks"], args.overlap_sigma,
                               args.overlap_sigma_scan or None)
        diagnostics[arm] = diag

        with open(os.path.join(out_dir, f"arm_{arm}_diagnostics.json"), "w") as f:
            json.dump(diag, f, indent=2)
        if args.save_index:
            index = build_index(dataset, res["indices"])
            with open(os.path.join(out_dir, f"arm_{arm}_index.json"), "w") as f:
                json.dump(index, f)
        if args.save_model:
            model = res["model"]
            if arm == "drq":
                torch.save({
                    "vae_state_dict": model.state_dict(),
                    "centroids": [c.tolist() for c in res["codebooks"]],
                }, os.path.join(out_dir, "arm_drq_model.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(out_dir, f"arm_{arm}_model.pth"))

        logger.info(f"[{arm}] injectivity diagnostics saved")

    # Per-arm summary tables
    for arm in arms:
        d = diagnostics[arm]
        res = results[arm]
        logger.info("")
        logger.info(f"--- {arm} reconstruction ---")
        log_table(logger, [
            ("val recon (continuous)", f"{res['recon_continuous']:.6f}", "decoder(encoder(x))"),
            ("val recon (quantized)", f"{res['recon_quantized']:.6f}", "decoder(code sum)"),
            ("quantization cost", f"{res['recon_quantized'] - res['recon_continuous']:.6f}",
             "quantized - continuous"),
        ], ("metric", "value", "note"))
        logger.info(f"--- {arm} tuple diagnostics ---")
        rows = [(k, f"{v:,}" if isinstance(v, int) else f"{v:.5f}", "")
                for k, v in d["tuples"].items()]
        log_table(logger, rows)

    # Comparison: each candidate arm against the rqvae baseline. Works when the
    # arms ran in-process, or as separate jobs sharing the same --run_name / output dir.
    diag_files = {
        "rqvae": os.path.join(out_dir, "arm_rqvae_diagnostics.json"),
        "warmup": os.path.join(out_dir, "arm_warmup_diagnostics.json"),
        "drq": os.path.join(out_dir, "arm_drq_diagnostics.json"),
    }
    if os.path.exists(diag_files["rqvae"]):
        base_diag = json.load(open(diag_files["rqvae"]))
        for cand in ("warmup", "drq"):
            if not os.path.exists(diag_files[cand]):
                continue
            cand_diag = json.load(open(diag_files[cand]))
            comparison = build_comparison(base_diag, cand_diag)
            cmp_name = "comparison.json" if cand == "drq" else f"comparison_{cand}_vs_rqvae.json"
            with open(os.path.join(out_dir, cmp_name), "w") as f:
                json.dump(comparison, f, indent=2)
            print_comparison(comparison, logger, cand_name=cand)
            _sigma_scan_table(base_diag, cand_diag, logger)

    logger.info(f"done. artifacts in {out_dir}")


if __name__ == "__main__":
    main()
