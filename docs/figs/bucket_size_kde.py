"""EDA figure: SID bucket-size distributions of the three real arms.

Classical RQ-VAE (no revival) vs Revival RQ-VAE vs DRQ, computed
from the actual assignment indexes (4,783,154 items, 3x1024 codes):

  A  archive/2026-11-14-broken/runs/sid_3x1024_broken/arm_rqvae_index.json
  B  archive/2026-11-07-baseline/runs/sid_3x1024_baseline/arm_rqvae_index.json
  C  archive/2026-11-07-drg/runs/sid_3x1024_baseline/arm_drq_index.json

All item-weighted panels restrict to COLLIDING items (bucket size >= 2):
the clean singletons (53-74% of each corpus) are the floor of the story, the
collision mass is the point. Three lenses on "DRQ owns the tail":

  (a) KDE of bucket size over log2 axis, colliding items only. Positive
      skew = tail; DRQ should fall off fastest, broken drift right.
  (b) Histogram: share of colliding items per bucket-size bin, log counts.
      The red bars past x=196 have no green counterpart at all.
  (c) Tail ownership: share of colliding items in buckets of size >= k.
      Curves that stay high at large k own the tail.

Bucket-size histograms are cached to bucket_size_counts.npz; delete it to
recompute from the indexes (~1 min/arm first run).

Run:  python bucket_size_kde.py   ->  bucket_size_kde.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CACHE = HERE / "bucket_size_counts.npz"
OUT = HERE / "bucket_size_kde.png"

ARMS = [
    ("Classical RQ-VAE",
     "archive/2026-11-14-broken/runs/sid_3x1024_broken/arm_rqvae_index.json", "#e53e3e"),
    ("Revival RQ-VAE",
     "archive/2026-11-07-baseline/runs/sid_3x1024_baseline/arm_rqvae_index.json", "#3182ce"),
    ("DRQ", "archive/2026-11-07-drg/runs/sid_3x1024_baseline/arm_drq_index.json", "#38a169"),
]
ROOT = HERE.parent.parent  # repo root (this file lives in docs/figs)


def bucket_counts_from_index(path):
    """h[s] = number of SID buckets with exactly s items (first run ~40 s/arm)."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    packed = np.fromiter(
        ((v[0] << 20) | (v[1] << 10) | v[2] for v in d.values()),
        dtype=np.int64, count=len(d),
    )
    del d
    _, inv = np.unique(packed, return_inverse=True)
    sizes = np.bincount(inv)
    h = np.bincount(sizes)
    print(f"  {path.parent.parent.parent.name}/{path.name}: {len(sizes)} buckets, "
          f"max {sizes.max()}, items {sizes.sum()}")
    return h


if CACHE.exists():
    h = {k: v for k, v in np.load(CACHE).items()}
else:
    print("computing bucket-size histograms from indexes...")
    h = {}
    for name, rel, _c in ARMS:
        h[name] = bucket_counts_from_index(ROOT / rel)
    np.savez_compressed(CACHE, **h)

names = [a[0] for a in ARMS]
colors = {n: c for n, _r, c in ARMS}
maxb = {n: len(h[n]) - 1 for n in names}
item_mass = {n: h[n] * np.arange(len(h[n])) for n in names}   # items in size-s buckets
coll_mass = {n: item_mass[n].copy() for n in names}
for n in names:
    coll_mass[n][1] = 0                                        # drop singletons
coll_tot = {n: coll_mass[n].sum() for n in names}
tot_items = {n: item_mass[n].sum() for n in names}


def kde_curve(mass, bw=0.18, n_grid=200):
    """Gaussian KDE of log2 bucket sizes from a weighted histogram."""
    sizes = np.arange(2, len(mass))
    w = mass[2:] / mass[2:].sum()
    lm = np.log2(sizes)
    grid = np.linspace(1.0, np.log2(len(mass) - 1), n_grid)
    dens = np.zeros(n_grid)
    for xi, wi in zip(lm, w):
        dens += wi * np.exp(-0.5 * ((grid - xi) / bw) ** 2)
    dens /= bw * np.sqrt(2 * np.pi)
    # transform density of log2(size) to density over size itself
    return 2 ** grid, dens / (2 ** grid * np.log(2.0))


fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

# ---- (a) KDE of bucket size, colliding items only -------------------------
ax = axes[0]
for n in names:
    xg, dens = kde_curve(coll_mass[n])
    ax.plot(xg, dens, color=colors[n], lw=2.2, label=n)
    ax.fill_between(xg, dens, color=colors[n], alpha=0.10)
ticks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
ax.set_xscale("log", base=2)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
ax.set_xlabel("items sharing this SID")
ax.set_ylabel("density (colliding items)")
ax.set_title("(a) shape of the tail: colliding items\nClassical = highest, flattest tail",
             fontweight="bold")
ax.legend(frameon=False, fontsize=9.5)

# ---- (b) histogram of bucket-size bins, colliding items -------------------
ax = axes[1]
edges = np.array([1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 12.5, 16.5, 32.5, 64.5, 128.5, 642])
centers = (edges[:-1] + edges[1:]) / 2
widths = np.diff(edges) / 3.2
for i, n in enumerate(names):
    m = coll_mass[n]
    if len(m) < len(edges):
        m = np.concatenate([m, np.zeros(len(edges) - len(m))])
    b = np.clip(np.digitize(np.arange(len(m)), edges) - 1, 0, len(centers) - 1)
    share = np.zeros(len(centers))
    np.add.at(share, b[2:], m[2:])
    share /= coll_tot[n]
    ax.bar(centers + (i - 1) * widths, share, width=widths, color=colors[n],
           alpha=0.85, label=n)
ax.set_xscale("log", base=2)
ax.set_xticks(edges[1:] - 0.5)
ax.set_xticklabels([str(int(e)) for e in edges[1:] - 0.5], fontsize=8.5)
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_xlabel("bucket size upper edge")
ax.set_ylabel("share of colliding items (log)")
ax.set_title("(b) histogram, same data as counts\nno green bars past 196",
             fontweight="bold")
ax.legend(frameon=False, fontsize=9.5)

# ---- (c) tail ownership curve ----------------------------------------------
ax = axes[2]
for n in names:
    ks = np.arange(2, maxb[n] + 1)
    tail = np.cumsum(coll_mass[n][len(coll_mass[n]) - 1:1:-1])[::-1] / coll_tot[n]
    kk = np.unique(np.round(np.exp(np.linspace(np.log(2), np.log(maxb[n]), 130))).astype(int))
    ax.plot(kk, [tail[min(k, len(tail)) - 2] for k in kk], color=colors[n], lw=2.2,
            label=f"{n}  (max bucket {maxb[n]})")
    ax.axvline(maxb[n], color=colors[n], ls=":", lw=1.2, alpha=0.7)
ax.set_xscale("log", base=2)
ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.set_xticklabels([2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.set_yscale("log")
ax.set_ylim(1e-4, 1.05)
ax.set_yticks([1, 0.1, 0.01, 0.001])
ax.set_yticklabels(["100%", "10%", "1%", "0.1%"])
ax.set_xlabel("k   (share of colliding items in buckets of size >= k)")
ax.set_ylabel("share of colliding items")
ax.set_title("(c) tail ownership\nleftmost curve at every k = smallest tail",
             fontweight="bold")
ax.legend(frameon=False, fontsize=9)

# ---- summary numbers under the figure --------------------------------------
def at_least(n, k):
    return coll_mass[n][k:].sum() / coll_tot[n]


stat_lines = []
for n in names:
    stat_lines.append(
        f"{n:32s} colliding items {coll_tot[n]/1e6:5.2f}M "
        f"({coll_tot[n]/tot_items[n]*100:4.1f}% of corpus)   "
        f"in buckets >=16: {at_least(n, 16)*100:4.1f}%   >=64: {at_least(n, 64)*100:4.1f}%   "
        f"max bucket {maxb[n]}"
    )
fig.text(0.5, -0.045, "from the real indexes:\n" + "\n".join(stat_lines),
         ha="center", fontsize=9.5, family="monospace", color="#222")

fig.suptitle("Real data, 4.78M items, 3x1024 codes: how much of the collision tail each arm owns",
             y=1.02, fontsize=13.5)
fig.subplots_adjust(bottom=0.14, top=0.85, wspace=0.14)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(OUT)
