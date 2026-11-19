"""EDA figure: number of distinct SIDs (tuples) with bucket size >= k.

log-log, colliding SIDs only, from bucket_size_counts.npz (extended with
the warmup arm from its index on first run).

Run:  python tuple_count_vs_size.py   ->  tuple_count_vs_size.png
"""
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CACHE = HERE / "bucket_size_counts.npz"
ROOT = HERE.parent.parent

ARMS = [
    ("Classical RQ-VAE", "#D64541"),
    ("Revival RQ-VAE", "#2E6FC2"),
    ("Warmup RQ-VAE", "#E5A33B"),
    ("DRQ", "#2FA36B"),
]
WARMUP_INDEX = "archive/2026-11-07-wae/runs/sid_3x1024_baseline/arm_warmup_index.json"

h = dict(np.load(CACHE))
if "Warmup RQ-VAE" not in h:
    path = ROOT / WARMUP_INDEX
    if not path.exists():  # fall back to the gzip on GitHub
        path = Path(str(path) + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    packed = np.fromiter(((v[0] << 20) | (v[1] << 10) | v[2] for v in d.values()),
                         dtype=np.int64, count=len(d))
    del d
    _, inv = np.unique(packed, return_inverse=True)
    h["Warmup RQ-VAE"] = np.bincount(np.bincount(inv))
    np.savez_compressed(CACHE, **h)
    print("added Warmup RQ-VAE to cache, max bucket", len(h["Warmup RQ-VAE"]) - 1)

fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=300)
for label, color in ARMS:
    hh = h[label]
    maxb = len(hh) - 1
    tail = np.cumsum(hh[2:][::-1])[::-1]  # SIDs of size >= k, k = 2..maxb
    ks = np.unique(np.round(np.exp(np.linspace(np.log(2), np.log(maxb), 160))).astype(int))
    ys = [tail[min(k, maxb) - 2] for k in ks]
    ax.plot(ks, ys, color=color, lw=2.4, label=f"{label}  (max {maxb})")
    ax.axvline(maxb, color=color, ls=":", lw=1.4, alpha=0.7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.set_xticklabels([2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.set_xlabel("k   (SIDs with bucket size >= k)", fontsize=15)
ax.set_ylabel("number of such SIDs", fontsize=15)
ax.set_title("Tail count by bucket size", fontsize=17, fontweight="bold")
ax.legend(frameon=False, fontsize=13)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(alpha=0.2)
ax.set_axisbelow(True)
ax.tick_params(labelsize=13)

fig.tight_layout()
out = HERE / "tuple_count_vs_size.png"
fig.savefig(out, dpi=300, facecolor="white")
print(out)
