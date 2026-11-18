"""Concept figure for the geometry penalty (DRQ's motivation, illustrated).

Three 3D panels sharing one view:
  (a) the data: a 2D manifold curved through 3D, stretched along one axis
      (anisotropy), with dense clumps (uneven item density);
  (b) RQ-VAE: reconstruction candidates form a rigid lattice (the sum of
      axis-aligned per-level codebooks is a grid). Almost every grid site
      strands off-manifold, and the dense clumps overload the few sites the
      sheet passes through -> collisions;
  (c) k-means / DRQ stage 2: same budget of codes, placed by the data.
      All codes land on the manifold and the clumps get split finer.

The data is synthetic on purpose: it shows the mechanism, not our 4.78M
TencentGR embeddings.

Run:  python manifold_vs_grid_3d.py   ->  manifold_vs_grid_3d.png
"""
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "manifold_vs_grid_3d.png"

N_ITEMS = 1500
N_CODES = 1024  # one level of the real codebook, for visual parity


def sample_manifold(n):
    """Roll a widening strip (curved), stretch x (anisotropy), clump density."""
    u = np.concatenate(
        [c + rng.normal(0, 0.25, n // 5) for c in (1.0, 3.6, 5.4)]
        + [rng.uniform(0, 2 * np.pi, n - 3 * (n // 5))]
    )
    v = rng.uniform(0, 1, len(u))
    r = 2.0 + 0.7 * v                            # strip widens along v
    x = 2.4 * r * np.cos(u)                      # anisotropic: x stretched
    y = r * np.sin(u)
    z = 0.9 * np.sin(1.6 * u) * v                # curvature couples u and v
    return np.stack([x, y, z], axis=1)


X = sample_manifold(N_ITEMS)
lo, hi = X.min(0) - 0.7, X.max(0) + 0.7


def lattice_candidates(k):
    """Regular 3D grid, ~k sites in the data box: what a fixed set of
    axis-aligned per-level codebooks can sum to (a rigid lattice)."""
    n = int(round(k ** (1 / 3)))
    axes = [np.linspace(lo[d], hi[d], n) for d in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    spacing = np.mean([axes[d][1] - axes[d][0] for d in range(3)])
    return G, spacing, axes


def draw_grid_lines(ax, axes, color="#94a3b8", lw=0.45, alpha=0.22):
    """Draw orthogonal wireframe grid lines to emphasize the rigid 3D lattice."""
    xs, ys, zs = axes
    for x in xs:
        for y in ys:
            ax.plot([x, x], [y, y], [zs[0], zs[-1]], color=color, lw=lw, alpha=alpha, zorder=0)
    for x in xs:
        for z in zs:
            ax.plot([x, x], [ys[0], ys[-1]], [z, z], color=color, lw=lw, alpha=alpha, zorder=0)
    for y in ys:
        for z in zs:
            ax.plot([xs[0], xs[-1]], [y, y], [z, z], color=color, lw=lw, alpha=alpha, zorder=0)


def nearest(points, centers):
    d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1), np.sqrt(d2.min(1))


def stats(labels, n_codes):
    counts = np.bincount(labels, minlength=n_codes)
    return counts, (counts - 1).clip(min=0).sum() / len(labels)


G, spacing, axes = lattice_candidates(N_CODES)
K = KMeans(N_CODES, n_init=3, random_state=0, max_iter=300, tol=1e-2).fit(
    X
).cluster_centers_

gl, gerr = nearest(X, G)
kl, kerr = nearest(X, K)
g_counts, g_collide = stats(gl, len(G))
k_counts, k_collide = stats(kl, len(K))
k_unused = (k_counts == 0).sum()
reachable = (nearest(G, X)[1] < 0.5 * spacing).sum()

ELEV, AZIM = 14, -60
fig = plt.figure(figsize=(16.5, 5.8))


def panel(title, sub):
    ax = fig.add_subplot(1, 3, len(fig.axes) + 1, projection="3d")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.dist = 11.0
    ax.text2D(0.0, 0.0, sub, transform=ax.transAxes, fontsize=9.5, color="#333")
    return ax


def pts(ax, P, c, s, a):
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=c, s=s, alpha=a, linewidths=0)


def codes(ax, counts, C):
    pts(ax, C[counts == 0], "#94a3b8", 8, 0.28)
    pts(ax, C[(counts >= 1) & (counts <= 2)], "#3182ce", 14, 0.95)
    hot = counts > 2
    pts(ax, C[hot], "#e53e3e", 7 + 4 * np.sqrt(counts[hot]), 0.95)


ax = panel("(a) the data: curved, anisotropic, clumped",
           "item embeddings live on a 2D manifold bent through 3D,\n"
           "stretched along x, with dense clumps (60% of items)")
pts(ax, X, "#2b6cb0", 6, 0.55)

ax = panel("(b) RQ-VAE: candidates form a rigid grid",
           f"grid aligned to the axes, not the manifold: only {reachable}/{len(G)} sites\n"
           f"sit near the sheet; {g_collide*100:.0f}% of items lose their ID; snap error {gerr.mean():.2f}")
draw_grid_lines(ax, axes)
pts(ax, X, "#a0aec0", 3, 0.25)
codes(ax, g_counts, G)

ax = panel("(c) k-means / DRQ stage 2: codes follow the data",
           f"same budget of codes, positioned by the data: {k_unused}/{len(K)} unused, "
           f"clumps\nsplit finer; {k_collide*100:.0f}% of items lose their ID; snap error {kerr.mean():.2f}")
pts(ax, X, "#a0aec0", 3, 0.25)
codes(ax, k_counts, K)

handles = [
    Line2D([], [], marker="o", ls="", ms=5, color="#a0aec0", label="item embeddings"),
    Line2D([], [], marker="o", ls="", ms=5, color="#94a3b8", label="code, no item nearby"),
    Line2D([], [], marker="o", ls="", ms=7, color="#3182ce", label="code, 1-2 items"),
    Line2D([], [], marker="o", ls="", ms=10, color="#e53e3e", label="collision bucket (3+ items)"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
fig.suptitle("Geometry penalty: a flat codebook lattice over a curved, anisotropic manifold",
             y=0.98, fontsize=13.5)
fig.subplots_adjust(bottom=0.13, top=0.85, wspace=0.02)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(OUT)