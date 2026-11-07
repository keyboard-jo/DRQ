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
              geometry cross-overlap.
  full tuple  unique tuples, collision rate, max collision bucket, joint
              entropy/perplexity, effective cells (birthday inversion),
              stage redundancy (total correlation).

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
BETA = 0.25               # paper commitment weight (TIGER/SoundStream)
LR = 5e-4
EPOCHS = 50
BATCH_SIZE = 4096
INFERENCE_BATCH_SIZE = 16384
VAL_RATIO = 0.1
WARMUP_STEPS = 2000
MIN_LR_RATIO = 0.01
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
OVERLAP_SIGMA = 0.1       # latent perturbation scale for the O_pi kernel
KMEANS_FIT_SAMPLES = 500_000
KMEANS_N_INIT = 3
EVAL_SUBSET_N = 200_000


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
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
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
            cb_loss = cb_loss + F.mse_loss(quantized, remainder.detach())
            cm_loss = cm_loss + F.mse_loss(remainder, quantized.detach())
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

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        q = self.quantizer(z)
        # Straight-through estimator: decoder gradient flows to z as-is.
        z_ste = z + (q["quantized_sum"] - z).detach()
        x_hat = self.decoder(z_ste)
        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss + q["cb_loss"] + self.beta * q["cm_loss"]
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "cb_loss": q["cb_loss"],
            "cm_loss": q["cm_loss"],
            "indices": q["indices"],
            "quantized_sum": q["quantized_sum"],
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


def hierarchical_kmeans(z_all: torch.Tensor, codebook_sizes: list[int],
                        logger: logging.Logger, fit_samples: int, n_init: int,
                        random_state: int, assign_batch_size: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """DRQ stage 2: residual k-means on frozen latents.

    Fits on a subsample (fast), assigns the whole corpus (torch matmul),
    recurses on residuals. Returns (centroids list, indices (N, L)).
    """
    n, d = z_all.shape
    fit_n = min(int(fit_samples), n)
    rng = np.random.default_rng(random_state)
    fit_idx = rng.choice(n, size=fit_n, replace=False) if fit_n < n else np.arange(n)

    residual_all = z_all
    centroids: list[torch.Tensor] = []
    idxs: list[torch.Tensor] = []
    for stage, k in enumerate(codebook_sizes):
        k_eff = min(int(k), fit_n)
        km = KMeans(n_clusters=k_eff, n_init=n_init, random_state=random_state,
                    max_iter=300, tol=1e-4)
        km.fit(residual_all[fit_idx].numpy())
        c = torch.from_numpy(km.cluster_centers_.astype(np.float32))
        centroids.append(c)
        labels = assign_nearest(residual_all, c, assign_batch_size)
        idxs.append(labels)
        residual_all = residual_all - c[labels]
        mean_sq = float((residual_all * residual_all).sum(dim=-1).mean().item())
        logger.info(f"    level {stage}: k-means K={k_eff} done, "
                    f"residual ||r||^2 mean = {mean_sq:.4f}")
    return centroids, torch.stack(idxs, dim=-1)


# --------------------------------------------------------------------------- #
# Diagnostics (DRQ injectivity metrics)
# --------------------------------------------------------------------------- #
def _overlap_metrics(counts: torch.Tensor, centroids: torch.Tensor,
                     sigma: float) -> tuple[float, float, float, float]:
    """O_pi, K_eff, distribution floor, geometry cross-overlap (per level)."""
    k = centroids.shape[0]
    total = counts.sum().item()
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    pi = counts.float() / total
    distribution_floor = float((pi * pi).sum().item())
    if sigma > 0:
        c = centroids.float()
        d2 = torch.cdist(c, c, p=2.0).pow(2.0)               # (K, K)
        w = torch.exp(-d2 / (4.0 * sigma * sigma))
    else:
        w = torch.eye(k, device=centroids.device, dtype=torch.float32)
    o_pi = float((pi @ w @ pi).item())
    k_eff = 1.0 / o_pi if o_pi > 0 else 0.0
    geom = o_pi - distribution_floor
    return o_pi, k_eff, distribution_floor, geom


def compute_stage_diagnostics(indices: torch.Tensor, codebooks: list[torch.Tensor],
                              sigma: float) -> list[dict]:
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
        o_pi, k_eff, dist_floor, geom = _overlap_metrics(counts, codebooks[stage], sigma)
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
            "distribution_floor": dist_floor,
            "overlap_rate": o_pi,
            "effective_codes": k_eff,
            "geometry_cross_overlap": geom,
        })
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
    collision_rate = items_in_collision / n if n else 0.0
    p = counts.float() / n
    joint_entropy = float(-(p * torch.log(p + 1e-12)).sum().item())
    joint_perplexity = math.exp(joint_entropy)
    return {
        "n_items": n,
        "tuple_length": l,
        "n_unique_tuples": n_unique,
        "uniqueness_ratio": (n_unique / n) if n else 0.0,
        "collision_rate": collision_rate,
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
                    sigma: float) -> dict:
    """Full injectivity report for one arm."""
    stage = compute_stage_diagnostics(indices, codebooks, sigma)
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
def evaluate_recon(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for (x,) in loader:
        x = x.to(device)
        if isinstance(model, ContinuousVAE):
            _, x_hat = model(x)
        else:
            x_hat = model.decoder(model.encoder(x))
        total += F.mse_loss(x_hat, x).item() * x.shape[0]
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
                         device: torch.device, logger: logging.Logger) -> ContinuousVAE:
    """DRQ stage 1: reconstruction only, no codebook."""
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = _make_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio)
    step = 0
    for epoch in range(epochs):
        vae.train()
        running = defaultdict(float)
        t0 = time.time()
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            z, x_hat = vae(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1
            running["loss"] += loss.item()
            if step % 100 == 0:
                logger.info(
                    f"    [DRQ stage1] epoch {epoch + 1}/{epochs} step {step} "
                    f"recon={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )
        val_recon = evaluate_recon(vae, val_loader, device)
        logger.info(
            f"  [DRQ stage1] epoch {epoch + 1}/{epochs} "
            f"train_recon={running['loss'] / max(1, steps_per_epoch):.4f} "
            f"val_recon={val_recon:.4f} ({time.time() - t0:.1f}s)"
        )
    return vae


def train_rqvae(model: RQVAE, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float, warmup_steps: int, min_lr_ratio: float,
                device: torch.device, logger: logging.Logger) -> RQVAE:
    """Baseline: joint geometry + codebook training through STE."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = _make_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio)
    k0, k1 = model.codebook_sizes[0], model.codebook_sizes[-1]
    step = 0
    for epoch in range(epochs):
        model.train()
        running = defaultdict(float)
        t0 = time.time()
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1
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
        val_recon = evaluate_recon(model, val_loader, device)
        n = max(1, steps_per_epoch)
        logger.info(
            f"  [RQVAE] epoch {epoch + 1}/{epochs} "
            f"loss={running['loss'] / n:.4f} recon={running['recon'] / n:.4f} "
            f"cb={running['cb'] / n:.4f} cm={running['cm'] / n:.4f} "
            f"val_recon={val_recon:.4f} ({time.time() - t0:.1f}s)"
        )
    return model


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def run_rqvae_arm(dataset: EmbeddingDataset, train_idx: np.ndarray, val_idx: np.ndarray,
                  codebook_sizes: list[int], args, logger: logging.Logger) -> dict:
    logger.info("=== Arm 1: baseline RQ-VAE (joint training) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RQVAE(args.input_dim, args.hidden_dim, codebook_sizes, args.beta).to(device)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    train_rqvae(model, train_loader, val_loader, args.epochs, args.lr,
                args.warmup_steps, args.min_lr_ratio, device, logger)
    indices = encode_indices_full(model, dataset.embeddings, args.inference_batch_size, device)
    codebooks = [cb.detach().cpu() for cb in model.quantizer.codebooks]
    return {"indices": indices, "codebooks": codebooks, "model": model}


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
    train_continuous_vae(vae, train_loader, val_loader, args.epochs, args.lr,
                         args.warmup_steps, args.min_lr_ratio, device, logger)
    # Stage 2: hierarchical k-means on frozen full-corpus latents
    z_all = encode_latents(vae, dataset.embeddings, args.inference_batch_size, device)
    centroids, indices = hierarchical_kmeans(
        z_all, codebook_sizes, logger, args.kmeans_fit_samples,
        args.kmeans_n_init, args.seed, args.inference_batch_size,
    )
    return {"indices": indices, "codebooks": centroids, "model": vae}


# --------------------------------------------------------------------------- #
# Output / comparison
# --------------------------------------------------------------------------- #
def build_index(dataset: EmbeddingDataset, indices: torch.Tensor) -> dict:
    return {
        item_id: [int(v) for v in row.tolist()]
        for item_id, row in zip(dataset.item_ids, indices)
    }


def build_comparison(rqvae_diag: dict, drq_diag: dict) -> dict:
    def stage_headline(diag: dict) -> dict:
        s0 = diag["stage"][0]
        return {
            "level0_perplexity": s0["perplexity"],
            "level0_gini": s0["gini"],
            "level0_effective_codes": s0["effective_codes"],
            "level0_overlap_rate": s0["overlap_rate"],
        }

    return {
        "rqvae": {
            "tuples": rqvae_diag["tuples"],
            "stage_headline": stage_headline(rqvae_diag),
        },
        "drq": {
            "tuples": drq_diag["tuples"],
            "stage_headline": stage_headline(drq_diag),
        },
    }


def print_comparison(comparison: dict, logger: logging.Logger) -> None:
    r = comparison["rqvae"]
    d = comparison["drq"]
    rt, dt = r["tuples"], d["tuples"]
    logger.info("")
    logger.info("=" * 78)
    logger.info("COMPARISON  (RQ-VAE baseline  vs  DRQ)")
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
              ("metric", "rqvae", "drq"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DRQ vs RQ-VAE injectivity ablation")
    p.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH,
                   help="parquet or h5 file of precomputed item embeddings")
    p.add_argument("--arm", type=str, choices=["rqvae", "drq", "both"], default="both")
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
    p.add_argument("--overlap_sigma", type=float, default=OVERLAP_SIGMA)
    p.add_argument("--kmeans_fit_samples", type=int, default=KMEANS_FIT_SAMPLES)
    p.add_argument("--kmeans_n_init", type=int, default=KMEANS_N_INIT)
    p.add_argument("--eval_subset_n", type=int, default=EVAL_SUBSET_N)
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

    arms = ["rqvae", "drq"] if args.arm == "both" else [args.arm]
    results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    for arm in arms:
        if arm == "rqvae":
            res = run_rqvae_arm(dataset, train_idx, val_idx, codebook_sizes, args, logger)
        else:
            res = run_drq_arm(dataset, train_idx, val_idx, codebook_sizes, args, logger)
        results[arm] = res

        diag = run_diagnostics(res["indices"], res["codebooks"], args.overlap_sigma)
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
                torch.save(model.state_dict(), os.path.join(out_dir, "arm_rqvae_model.pth"))

        logger.info(f"[{arm}] injectivity diagnostics saved")

    # Per-arm summary tables
    for arm in arms:
        d = diagnostics[arm]
        logger.info("")
        logger.info(f"--- {arm} tuple diagnostics ---")
        rows = [(k, f"{v:,}" if isinstance(v, int) else f"{v:.5f}", "")
                for k, v in d["tuples"].items()]
        log_table(logger, rows)

    if "rqvae" in diagnostics and "drq" in diagnostics:
        comparison = build_comparison(diagnostics["rqvae"], diagnostics["drq"])
        with open(os.path.join(out_dir, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        print_comparison(comparison, logger)

    logger.info(f"done. artifacts in {out_dir}")


if __name__ == "__main__":
    main()
