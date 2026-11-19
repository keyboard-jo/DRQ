"""Simple PPT bar chart: relative reduction claimed by the DRQ paper vs ours.

  metric                paper      ours
  collision rate        -67%       -46.4%
  max bucket size       -54%       -69.4%

Run:  python paper_vs_ours_reduction.py   ->  paper_vs_ours_reduction.png
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

groups = ["collision rate\nreduction", "max bucket\nreduction"]
paper = [67.0, 54.0]
ours = [46.4, 69.0]

x = np.arange(len(groups))
w = 0.32

fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)
b1 = ax.bar(x - w / 2, paper, width=w, color="#8A94A6", label="DRQ paper")
b2 = ax.bar(x + w / 2, ours, width=w, color="#2FA36B", label="Ours")
for bars, vals in ((b1, paper), (b2, ours)):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"\u2212{v:.1f}%",
                ha="center", fontsize=16, color="#222")

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=16)
ax.set_ylabel("reduction (%)", fontsize=16)
ax.set_ylim(0, 84)
ax.tick_params(axis="y", labelsize=14)
ax.legend(fontsize=15, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()
out = HERE / "paper_vs_ours_reduction.png"
fig.savefig(out, dpi=300, facecolor="white")
print(out)
