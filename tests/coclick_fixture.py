"""Synthetic digr_data-shaped fixture for build_coclick_pairs.py tests,
plus brute-force ground truth (directed co-click counts in catalog-row space).
"""
import json
import os

import numpy as np

SIZES = [8, 8, 8]
OFFS = [0, 8, 64]
L = 3
SEP = 100
N_ITEMS = 300
N_ROWS = 40
F_FEATS = 2
SEED = 0


def build(root: str):
    """Create root/{vocab.json, catalog_sid_codes.npy, n_items.txt, ids.txt,
    train/*} and return (n_items, truth) with truth a {(a,b): count} dict
    over catalog-row indices, including last-event->target anchor pairs."""
    rng = np.random.default_rng(SEED)
    tuples = set()
    while len(tuples) < N_ITEMS:
        tuples.add(tuple(int(v) for v in rng.integers(0, 8, size=L)))
    codes = np.array(sorted(tuples), dtype=np.int64)
    n_items = len(codes)
    keymap = {tuple(int(v) for v in row): i
              for i, row in enumerate(codes.tolist())}

    toks, offs_, targets = [], [0], []
    for _ in range(N_ROWS):
        m = int(rng.integers(2, 7))
        row = list(rng.integers(0, SEP, size=F_FEATS)) + [SEP]
        for _ in range(m):
            item = int(rng.integers(0, n_items))
            row += [int(rng.integers(0, SEP))] + (codes[item] + np.array(OFFS)).tolist()
        toks += row
        offs_.append(len(toks))
        targets.append(int(rng.integers(0, n_items)))
    # plant exact duplicate pairs: repeat row 0 verbatim (same target too)
    toks += toks[offs_[0]:offs_[1]]
    offs_.append(len(toks))
    targets.append(targets[0])

    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    np.save(os.path.join(root, "train/tokens.npy"), np.array(toks, dtype=np.int64))
    np.save(os.path.join(root, "train/offsets.npy"), np.array(offs_, dtype=np.int64))
    np.save(os.path.join(root, "train/targets.npy"), np.array(targets, dtype=np.int64))
    np.save(os.path.join(root, "catalog_sid_codes.npy"), codes)
    json.dump({"sid_sizes": SIZES, "sid_offsets": OFFS, "num_levels": L,
               "sep_id": SEP}, open(os.path.join(root, "vocab.json"), "w"))
    json.dump({"n_items": n_items}, open(os.path.join(root, "n_items.txt"), "w"))
    with open(os.path.join(root, "ids.txt"), "w") as f:
        f.write("\n".join(f"item_{i}" for i in range(n_items)))

    truth = {}
    for r in range(len(offs_) - 1):
        body = toks[offs_[r] + F_FEATS + 1:offs_[r + 1]]
        ev = [tuple(int(v) - OFFS[j] for j, v in
                    enumerate(body[i * (L + 1) + 1:i * (L + 1) + 1 + L]))
              for i in range(len(body) // (L + 1))]
        mapped = [keymap[e] for e in ev]
        for a, b in zip(mapped, mapped[1:]):
            truth[(a, b)] = truth.get((a, b), 0) + 1
        truth[(mapped[-1], targets[r])] = truth.get((mapped[-1], targets[r]), 0) + 1
    return n_items, truth
