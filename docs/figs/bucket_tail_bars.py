"""Simple PPT bar chart: items in bucket sizes above 16, out to 641.

Reuses bucket_size_counts.npz from bucket_size_kde.py (delete and re-run
that script to rebuild it from the real indexes).

Run:  python bucket_tail_bars.py   ->  bucket_tail_bars.png
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
h = dict(np.load(HERE / "bucket_size_counts.npz"))

KEYS = list(h.keys())  # full arm names as stored in the cache
ARM_ORDER = [("classical", "Classical RQ-VAE", "#D64541"),
             ("revival", "Revival RQ-VAE", "#2E6FC2"),
             ("drq", "DRQ", "#2FA36B")]
matched = []
for needle, label, color in ARM_ORDER:
    key = next(k for k in KEYS if needle in k.lower())
    matched.append((key, label, color))

BINS = [(17, 32), (33, 64), (65, 128), (129, 256), (257, 641)]

counts = []
for key, _label, _c in matched:
    mass = h[key] * np.arange(len(h[key]))   # items in buckets of size s
    counts.append([mass[lo:min(hi, len(mass) - 1) + 1].sum() / 1e6 for lo, hi in BINS])

x = np.arange(len(BINS))
w = 0.26

fig, ax = plt.subplots(figsize=(11.5, 6), dpi=300)
for i, ((_key, label, color), c) in enumerate(zip(matched, counts)):
    bars = ax.bar(x + i * w - w, c, width=w, color=color, label=label)
    for b, v in zip(bars, c):
        text = "0" if v == 0 else ("<0.01" if v < 0.005 else f"{v:.2f}")
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, text,
                ha="center", fontsize=13, color="#222")

ax.set_xticks(x)
ax.set_xticklabels([f"{lo}\u2013{hi}" for lo, hi in BINS], fontsize=17)
ax.set_xlabel("SID bucket size", fontsize=18)
ax.set_ylabel("items (millions)", fontsize=18)
ax.tick_params(axis="y", labelsize=15)
ax.set_ylim(0, max(max(c) for c in counts) * 1.14)
ax.legend(fontsize=16, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()
out = HERE / "bucket_tail_bars.png"
fig.savefig(out, dpi=300, facecolor="white")
print(out)
