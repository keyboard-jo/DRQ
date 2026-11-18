#!/usr/bin/env python3
"""Build the co-click co-occurrence matrix from the DiGR data layout and
export it as (1) a compressed sparse CSR npz and (2) a positive-pairs file
usable as --drq_pairs / --rqvae_lambda input in train_drq_vs_rqvae.py.

Machinery adapted from references/baselines_digr.py (build_layout /
iter_code_chunks / CodeIndex / build_coclick); evaluation, tie-break and
checkpoint code were dropped. CPU only, numpy + pyarrow (parquet export
falls back to csv.gz).

Data layout expected in --data_dir:
  vocab.json            {"sid_sizes": [..], "sid_offsets": [..],
                         "num_levels": L, "sep_id": int}
  catalog_sid_codes.npy [n_items, L] int64, row = catalog item index
  <split>/tokens.npy    flat int64 event tokens
  <split>/offsets.npy   [n_rows+1] int64 row bounds into tokens
  <split>/targets.npy   [n_rows] int64 catalog row index per row

Each row: [feature tokens][SEP][(temporal token, SID_1..SID_L) * m].
A co-click is two items appearing in consecutive events of the same row;
optionally the last event -> target pair (same convention as the DiGR
next-item task) is included. Counts are aggregated over all rows.

Outputs:
  <out_prefix>.coclick.npz   CSR: indptr (int64, n_items+1), next_item
                             (int32), count (int32) + meta json sidecar.
                             Indexed by CATALOG ROW.
  <out_prefix>.pairs.parquet columns item_a, item_b, count -- one row per
                             unordered pair (directed a->b and b->a merged),
                             self-pairs dropped. Ids are catalog rows unless
                             --item_ids_file maps row -> item_id, in which
                             case the ids match the embeddings parquet and
                             the file can be fed to train_drq_vs_rqvae.py
                             directly.

Run on the cluster (data lives there), e.g.:
  python references/build_coclick_pairs.py \
      --data_dir /scratch/mscbdtspod2026/jksulaiman/data/digr_data \
      --item_ids_file /scratch/mscbdtspod2026/jksulaiman/data/item_embeddings.parquet \
      --out_prefix runs/coclick \
      --min_count 2 --max_pairs 20000000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_CHUNK_ROWS = 200_000


def setup_logger(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger("coclick")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][coclick] %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def load_vocab(data_dir: str, logger: logging.Logger) -> dict:
    path = os.path.join(data_dir, "vocab.json")
    with open(path) as f:
        vocab = json.load(f)
    sizes = vocab["sid_sizes"]
    num_levels = vocab.get("num_levels", len(sizes))
    if len(sizes) != num_levels:
        raise ValueError(f"vocab.json num_levels={num_levels} but sid_sizes has "
                         f"{len(sizes)} entries")
    if "sep_id" not in vocab:
        raise ValueError("vocab.json lacks 'sep_id'")
    logger.info(f"vocab: num_levels={num_levels} sid_sizes={sizes} sep_id={vocab['sep_id']}")
    return vocab


def load_catalog_codes(data_dir: str, logger: logging.Logger) -> np.ndarray:
    path = os.path.join(data_dir, "catalog_sid_codes.npy")
    codes = np.load(path)
    if codes.ndim != 2:
        raise ValueError(f"catalog_sid_codes.npy must be [n_items, L], got {codes.shape}")
    logger.info(f"catalog: {codes.shape[0]} items, {codes.shape[1]} levels")
    return codes


def load_targets(split_dir: str, logger: logging.Logger) -> np.ndarray:
    path = os.path.join(split_dir, "targets.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"split dir {split_dir} lacks targets.npy")
    t = np.load(path)
    if t.ndim != 1:
        raise ValueError(f"targets.npy must be 1D, got shape {t.shape}")
    return t


def pack_codes(codes: np.ndarray, sizes: list[int]) -> np.ndarray:
    """Mixed-radix tuple -> unique int64 key (same as baselines_digr.py)."""
    revcum = np.cumprod([1] + sizes[::-1])[::-1][1:]
    return codes.astype(np.int64) @ revcum.astype(np.int64)


class CodeIndex:
    """Invertible exact-SID lookup over the catalog (assumes injective tuples)."""

    def __init__(self, keys: np.ndarray, rows: np.ndarray, sizes: list[int]) -> None:
        self.keys = keys
        self.rows = rows
        self.sizes = sizes

    def lookup(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = pack_codes(codes, self.sizes)
        pos = np.searchsorted(self.keys, q)
        pos_c = np.clip(pos, 0, len(self.keys) - 1)
        found = self.keys[pos_c] == q
        return np.where(found, self.rows[pos_c], -1), found


def build_code_index(catalog_codes: np.ndarray, vocab: dict,
                     logger: logging.Logger) -> CodeIndex:
    sizes = vocab["sid_sizes"]
    keys = pack_codes(catalog_codes, sizes)
    uniq, rows = np.unique(keys, return_index=True)
    expected = int(np.prod(sizes))
    if len(uniq) != len(keys):
        dup = len(keys) - len(uniq)
        raise ValueError(f"catalog contains {dup} duplicate SID tuples; co-clicks "
                         "could not be mapped back to items uniquely")
    if expected <= len(keys):
        logger.info(f"code index: {len(uniq):,} unique tuples (non-injective domain, "
                    f"full k-index skipped)")
    else:
        logger.info(f"code index: {len(uniq):,} unique tuples, injective "
                    f"({len(keys):,} of {expected:,} cells used)")
    return CodeIndex(uniq, rows, sizes)


def row_feature_lengths(tokens: np.ndarray, offsets: np.ndarray, sep_id: int,
                        logger: logging.Logger) -> np.ndarray:
    """Token count before the SEP for each row (same logic as
    baselines_digr.row_feature_lengths: exactly one SEP per row, so the
    k-th SEP position belongs to row k)."""
    n_rows = len(offsets) - 1
    if n_rows <= 0:
        return np.zeros(0, dtype=np.int64)
    pos = np.flatnonzero(np.asarray(tokens) == int(sep_id)).astype(np.int64)
    if len(pos) != n_rows:
        raise ValueError(
            f"found {len(pos)} SEP tokens but expected one per row "
            f"({n_rows} rows); the split is not in the expected layout")
    feat_len = pos - offsets[:-1]
    if (feat_len < 0).any():
        raise ValueError("SEP appears before its row start")
    return feat_len.astype(np.int64)


class SplitLayout:
    def __init__(self, tokens, offsets, feat_len, n_events, base0, num_levels,
                 sep_id, sid_offsets, sid_sizes):
        self.tokens = tokens
        self.offsets = offsets
        self.feat_len = feat_len
        self.n_events = n_events
        self.base0 = base0
        self.stride = 1 + num_levels
        self.sep_id = sep_id
        self.sid_offsets = np.asarray(sid_offsets, dtype=np.int64)
        self.sid_sizes = np.asarray(sid_sizes, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.n_events)


def build_layout(data_dir: str, split: str, vocab: dict, mmap: bool = True,
                 logger: logging.Logger = None) -> SplitLayout:
    logger = logger or logging.getLogger("coclick")
    d = os.path.join(data_dir, split)
    tokens = np.load(os.path.join(d, "tokens.npy"), mmap_mode="r" if mmap else None)
    offsets = np.load(os.path.join(d, "offsets.npy"))
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f"{d}: tokens.npy dtype {tokens.dtype} is not integer")
    if offsets[0] != 0 or offsets[-1] != len(tokens):
        raise ValueError(f"{d}: offsets[0]={offsets[0]} offsets[-1]={offsets[-1]} "
                         f"len(tokens)={len(tokens)} -- not a full dense layout")
    n_rows = len(offsets) - 1
    feat_len = row_feature_lengths(tokens, offsets, vocab["sep_id"], logger)
    row_len = np.diff(offsets)
    stride = 1 + vocab["num_levels"]
    event_tokens = row_len - 1 - feat_len          # body AFTER the SEP
    if (event_tokens < 0).any() or (event_tokens % stride != 0).any():
        bad = np.flatnonzero((event_tokens < 0) | (event_tokens % stride != 0))[:4]
        raise ValueError(
            f"{d}: row body is not a whole number of (temporal, SID_1..SID_L) events; "
            f"example rows {bad.tolist()} row_len={row_len[bad].tolist()} "
            f"feat_len={feat_len[bad].tolist()} stride={stride}")
    n_events = event_tokens // stride
    base0 = offsets[:n_rows] + feat_len            # position of the SEP token
    logger.info(f"split {split}: {n_rows:,} rows, {int(n_events.sum()):,} events, "
                f"{(n_events > 0).sum():,} non-empty rows")
    return SplitLayout(tokens, offsets, feat_len, n_events, base0,
                       vocab["num_levels"], vocab["sep_id"],
                       vocab.get("sid_offsets", [0] * vocab["num_levels"]),
                       vocab["sid_sizes"])


def iter_event_chunks(layout: SplitLayout, chunk_rows: int):
    """Yield (owner, seq, codes) per row-chunk.

    owner[i] = local row index of event i (relative to chunk start),
    seq[i]   = event position within its row, codes[i] = (L,) SID codes.
    Whole rows per chunk, so consecutive-event pairs never cross chunks.
    """
    n = len(layout)
    tokens = layout.tokens
    for s in range(0, n, chunk_rows):
        e = min(s + chunk_rows, n)
        counts = layout.n_events[s:e].astype(np.int64)
        total = int(counts.sum())
        owner = (s + np.repeat(np.arange(e - s, dtype=np.int64), counts)).astype(np.int64)
        seq = np.arange(total, dtype=np.int32) - np.repeat(
            np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        starts = layout.base0[s:e]
        pos = (np.repeat(starts, counts) + seq * layout.stride).astype(np.int64)
        cols = 2 + np.arange(layout.stride - 1, dtype=np.int64)  # skip temporal
        raw = np.asarray(tokens[(pos[:, None] + cols[None, :])]).astype(np.int64)
        # tokens hold GLOBAL vocab ids (sid_offsets[level] + per-level code);
        # catalog codes are per-level -- subtract the offsets (same as
        # baselines_digr.iter_code_chunks)
        codes = raw - layout.sid_offsets[None, :]
        if codes.size:
            lo = codes.min(axis=0)
            hi = codes.max(axis=0)
            if (lo < 0).any() or (hi >= layout.sid_sizes).any():
                raise ValueError(
                    f"SID codes out of range after vocab-offset subtraction: "
                    f"min={lo.tolist()} max={hi.tolist()} sizes={layout.sid_sizes.tolist()} "
                    "-- catalog and split vocab disagree")
        yield owner, seq, codes


def build_coclick(layout: SplitLayout, vocab: dict, catalog_codes: np.ndarray,
                  index: CodeIndex, targets: np.ndarray, n_items: int,
                  logger: logging.Logger, chunk_rows: int,
                  with_anchor_pairs: bool):
    """Co-click counts in catalog-row space (CSR). Same semantics as
    baselines_digr.build_coclick: consecutive mapped events within a row,
    optionally plus the last-event -> target anchor pair."""
    t0 = time.time()
    key_parts = []
    n_ev_tot = 0
    unmapped = 0
    rows_with_pairs = 0
    for owner, seq, codes in iter_event_chunks(layout, chunk_rows):
        n_ev_tot += len(codes)
        rows, found = index.lookup(codes)
        if not found.all():
            if unmapped == 0:
                ex = codes[~found][:3].tolist()
                logger.warning(f"first unmapped event SIDs (SID -> catalog row "
                               f"failed; history holds items outside the catalog): {ex}")
            unmapped += int((~found).sum())
        rows = np.where(found, rows, -1).astype(np.int64)
        same_row = owner[:-1] == owner[1:]
        consecutive = seq[1:] == seq[:-1] + 1
        mask = same_row & consecutive
        p, q = rows[:-1][mask], rows[1:][mask]
        ok = (p >= 0) & (q >= 0)
        if ok.any():
            key_parts.append(p[ok] * n_items + q[ok])
            rows_with_pairs += int(np.unique(owner[:-1][mask][ok]).size)
        if with_anchor_pairs and len(codes):
            last_idx = np.flatnonzero(np.r_[owner[1:] != owner[:-1], True])
            last_rows = rows[last_idx]
            tg = targets[owner[last_idx]].astype(np.int64)
            ok2 = (last_rows >= 0) & (tg >= 0) & (tg < n_items)
            if ok2.any():
                key_parts.append(last_rows[ok2] * n_items + tg[ok2])
    logger.info(f"co-click scan: {n_ev_tot:,} events, {rows_with_pairs:,} rows with "
                f"adjacent pairs, {unmapped:,} unmapped events, {time.time() - t0:.1f}s")
    if not key_parts:
        raise ValueError("no co-click pairs found -- nothing to build")
    all_keys = np.concatenate(key_parts)
    keys, counts = np.unique(all_keys, return_counts=True)
    del all_keys
    prev_item = keys // np.int64(n_items)   # non-decreasing (keys sorted)
    indptr = np.searchsorted(prev_item, np.arange(n_items + 1, dtype=np.int64))
    table = {
        "indptr": indptr.astype(np.int64),
        "next_item": (keys % n_items).astype(np.int32),
        "count": counts.astype(np.int32),
    }
    nnz = len(table["next_item"])
    logger.info(f"co-click table: {nnz:,} distinct directed pairs, "
                f"{int(table['count'].sum()):,} total co-clicks, mean successors "
                f"{nnz / max(1, int((np.diff(table['indptr']) > 0).sum())):.2f}")
    return table


def load_row_ids(path: str, n_items: int, logger: logging.Logger) -> np.ndarray:
    """Catalog row -> item_id strings. Accepts a .json list, parquet/csv with
    an item_id column (row order = catalog order), or a plain text file with
    one id per line (row order = catalog order)."""
    if path.endswith(".json"):
        with open(path) as f:
            ids = json.load(f)
        ids = np.asarray(ids, dtype=object)
    elif path.endswith(".parquet") or path.endswith(".csv"):
        import pandas as pd
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        if "item_id" not in df.columns:
            raise ValueError(f"{path}: expected an 'item_id' column")
        ids = df["item_id"].astype(str).to_numpy()
    else:
        with open(path) as f:
            ids = np.asarray([ln.rstrip("\n") for ln in f], dtype=object)
    if len(ids) != n_items:
        raise ValueError(f"id file has {len(ids)} rows but catalog has {n_items} items; "
                         "row order must match catalog_sid_codes.npy")
    logger.info(f"mapped catalog rows -> item ids via {path}")
    return ids


def export_pairs(table: dict, n_items: int, out_path: str, min_count: int,
                 max_pairs: int | None, ids: np.ndarray | None,
                 logger: logging.Logger) -> None:
    rows = np.repeat(np.arange(n_items, dtype=np.int64), np.diff(table["indptr"]))
    cols = table["next_item"].astype(np.int64)
    cnt = table["count"].astype(np.int64)
    keep = (rows != cols) & (cnt >= min_count)          # drop self-pairs
    a, b, c = rows[keep], cols[keep], cnt[keep]
    lo, hi = np.minimum(a, b), np.maximum(a, b)          # merge directions
    keys = lo * n_items + hi
    uk, inv = np.unique(keys, return_inverse=True)
    agg = np.bincount(inv, weights=c).astype(np.int64)
    if max_pairs is not None and len(uk) > max_pairs:
        sel = np.argsort(-agg, kind="stable")[:max_pairs]
        sel = np.sort(sel)
        logger.info(f"capped to top {max_pairs:,} pairs by count "
                    f"(cut at count >= {agg[sel[-1]]})")
        uk, agg = uk[sel], agg[sel]
    ia, ib = uk // n_items, uk % n_items
    if ids is not None:
        ia, ib = ids[ia], ids[ib]
    import pandas as pd
    df = pd.DataFrame({"item_a": ia, "item_b": ib, "count": agg})
    if out_path.endswith(".parquet"):
        try:
            df.to_parquet(out_path, index=False)
        except ImportError:
            out_path = out_path.rsplit(".", 1)[0] + ".csv.gz"
            logger.warning(f"pyarrow missing; writing {out_path} instead")
            df.to_csv(out_path, index=False, compression="gzip")
    else:
        df.to_csv(out_path, index=False,
                  compression="gzip" if out_path.endswith(".gz") else None)
    logger.info(f"wrote {len(df):,} unordered pairs (min_count={min_count}) -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="co-click matrix + positive pairs builder")
    ap.add_argument("--data_dir", type=str, required=True, help="digr_data directory")
    ap.add_argument("--train_split", type=str, default=DEFAULT_TRAIN_SPLIT)
    ap.add_argument("--out_prefix", type=str, default="coclick")
    ap.add_argument("--min_count", type=int, default=1,
                    help="pairs export: minimum merged (unordered) co-click count")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="pairs export: keep top-N pairs by count")
    ap.add_argument("--pairs_ext", type=str, default="parquet", choices=["parquet", "csv.gz"])
    ap.add_argument("--item_ids_file", type=str, default=None,
                    help="json list, parquet/csv with item_id column, or one-id-per-line "
                         "text file in catalog row order; without it pair ids are catalog "
                         "row indices")
    ap.add_argument("--no_anchor_pairs", action="store_true",
                    help="exclude last-event -> target pairs (same convention as "
                         "baselines_digr.py training counts)")
    ap.add_argument("--chunk_rows", type=int, default=DEFAULT_CHUNK_ROWS)
    args = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logger(args.out_prefix + ".log")
    vocab = load_vocab(args.data_dir, logger)
    catalog = load_catalog_codes(args.data_dir, logger)
    n_items = len(catalog)
    if catalog.shape[1] != vocab["num_levels"]:
        raise ValueError(f"catalog codes {catalog.shape[1]} != num_levels "
                         f"{vocab['num_levels']}")
    index = build_code_index(catalog, vocab, logger)
    layout = build_layout(args.data_dir, args.train_split, vocab, True, logger)
    targets = load_targets(os.path.join(args.data_dir, args.train_split), logger)
    if len(targets) != len(layout):
        raise ValueError(f"targets.npy has {len(targets)} rows but split has "
                         f"{len(layout)} rows")

    table = build_coclick(layout, vocab, catalog, index, targets, n_items, logger,
                          args.chunk_rows, not args.no_anchor_pairs)
    np.savez_compressed(args.out_prefix + ".coclick.npz",
                        indptr=table["indptr"], next_item=table["next_item"],
                        count=table["count"])
    nnz = len(table["next_item"])
    with open(args.out_prefix + ".coclick.meta.json", "w") as f:
        json.dump({
            "n_items": int(n_items),
            "directed_pairs": int(nnz),
            "total_coclicks": int(table["count"].sum()),
            "max_count": int(table["count"].max()),
            "mean_successors": float(nnz / max(1, np.count_nonzero(np.diff(table["indptr"])))),
            "anchor_pairs": not args.no_anchor_pairs,
            "train_split": args.train_split,
            "note": "indexed by catalog row of catalog_sid_codes.npy",
        }, f, indent=2)
    logger.info(f"wrote {args.out_prefix}.coclick.npz + meta json")

    ids = load_row_ids(args.item_ids_file, n_items, logger) if args.item_ids_file else None
    export_pairs(table, n_items, f"{args.out_prefix}.pairs.{args.pairs_ext}",
                 args.min_count, args.max_pairs, ids, logger)
    logger.info("done")


if __name__ == "__main__":
    main()
