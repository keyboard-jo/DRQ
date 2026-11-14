#!/usr/bin/env python
"""H4 duplicate-floor scan: measure the injectivity floor of an embedding corpus.

Exact-duplicate embeddings (byte-identical float32 rows, or identical after
--round decimals) can never be separated by any quantizer: deterministic
nearest-centroid assignment maps every copy to the same SID bucket. This
script measures how much of the observed collision mass is that irreducible
floor, i.e. the ceiling on what any tokenizer change (DRQ included) can win.

CPU only. Needs: pyarrow, numpy, pandas. Streams the parquet in row groups;
RAM use is digests + ids, not the raw 256-dim array (4.78M x 256 x 4B ~ 4.9GB).

Usage:
  python scan_duplicate_floor.py --input_path /path/item_embeddings.parquet
  python scan_duplicate_floor.py --input_path ... \
      --sid_index runs/sid_3x1024_baseline/arm_drq_index.json --out h4_report.json
"""
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DIGEST = 16  # blake2b digest bytes per row; collisions between distinct rows ~ 2^-128


def detect_columns(schema, embedding_col, item_col):
    names = schema.names
    if embedding_col is None:
        for cand in ("embedding", "emb", "embeddings"):
            if cand in names:
                embedding_col = cand
                break
    if item_col is None:
        for cand in ("item_id", "id", "cid"):
            if cand in names:
                item_col = cand
                break
    if embedding_col is None or item_col is None:
        sys.exit(f"could not auto-detect columns in {names}; pass --embedding_col/--item_id_col")
    return item_col, embedding_col


def row_digests(arr):
    """blake2b of each row's float32 bytes. arr: (n, d) float32."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    void_t = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    v = a.view(void_t).reshape(-1)
    dig = np.empty(a.shape[0], dtype=object)
    for i in range(a.shape[0]):
        dig[i] = hashlib.blake2b(v[i].tobytes(), digest_size=DIGEST).digest()
    return dig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input_path", required=True, help="parquet: item_id + embedding (list<float>)")
    ap.add_argument("--sid_index", default=None,
                    help="optional arm_*_index.json {item_id: [c0,c1,...]} from a trained tokenizer; "
                         "splits collision mass into irreducible (duplicates) vs reducible")
    ap.add_argument("--embedding_col", default=None)
    ap.add_argument("--item_id_col", default=None)
    ap.add_argument("--round", type=int, default=None, metavar="DECIMALS",
                    help="round embeddings to N decimals before hashing (near-duplicate check; "
                         "exact-fp32 match can undercount if embedding gen was stochastic)")
    ap.add_argument("--row_group_rows", type=int, default=250_000)
    ap.add_argument("--top", type=int, default=10, help="print N largest duplicate groups")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()

    pf = pq.ParquetFile(args.input_path)
    item_col, emb_col = detect_columns(pf.schema_arrow, args.embedding_col, args.item_id_col)
    total = pf.metadata.num_rows
    print(f"[scan] {args.input_path}: {total:,} rows, embedding col '{emb_col}', id col '{item_col}'", flush=True)

    t0 = time.time()
    digests = np.empty(total, dtype=object)
    ids = np.empty(total, dtype=object)
    dim = None
    off = 0
    for batch in pf.iter_batches(batch_size=args.row_group_rows, columns=[item_col, emb_col]):
        n = batch.num_rows
        embs = np.stack([np.asarray(x, dtype=np.float32) for x in batch.column(emb_col).to_pylist()])
        if dim is None:
            dim = embs.shape[1]
        if args.round is not None:
            embs = np.round(embs, args.round)
        digests[off:off + n] = row_digests(embs)
        ids[off:off + n] = np.asarray(batch.column(item_col).to_pylist(), dtype=object).astype(str)
        off += n
        print(f"  [scan] {off:,}/{total:,} rows ({time.time()-t0:.0f}s)", flush=True)
    print(f"[scan] hashed {off:,} rows, dim={dim}, in {time.time()-t0:.1f}s", flush=True)

    # group by digest
    df = pd.DataFrame({"digest": digests, "item_id": ids})
    counts = df.groupby("digest", sort=False).size()
    dup_groups = counts[counts >= 2]
    dup_items = int(dup_groups.sum())
    dup_pairs = int((dup_groups * (dup_groups - 1) // 2).sum())
    n_distinct = int(len(counts))

    print("\n=== exact-duplicate scan ===")
    print(f"items:                 {off:,}")
    print(f"distinct embeddings:   {n_distinct:,}")
    print(f"duplicate groups:      {len(dup_groups):,}")
    print(f"items in dup groups:   {dup_items:,}  ({dup_items/max(off,1):.4%} of corpus)")
    print(f"duplicate pairs:       {dup_pairs:,}")
    if len(dup_groups):
        sizes = dup_groups.sort_values(ascending=False)
        print(f"largest group size:    {int(sizes.iloc[0]):,}")
        print(f"size distribution:     " + ", ".join(
            f">={t}: {int((sizes >= t).sum()):,}" for t in (2, 3, 5, 10, 50, 100, 500, 1000) if (sizes >= t).any()))

    report = {
        "input_path": args.input_path, "n_rows": int(off), "dim": dim,
        "round_decimals": args.round, "n_distinct": n_distinct,
        "dup_groups": int(len(dup_groups)), "items_in_dup_groups": dup_items,
        "dup_pairs": dup_pairs, "max_group_size": int(dup_groups.max()) if len(dup_groups) else 1,
        "group_size_hist": {str(k): int(v) for k, v in dup_groups.value_counts().items()} if len(dup_groups) else {},
    }

    if args.sid_index:
        t0 = time.time()
        with open(args.sid_index) as f:
            index = json.load(f)
        sid = pd.Series({k: tuple(v) for k, v in index.items()})
        n_tok = len(sid)
        uniq = int(sid.nunique())
        surplus = n_tok - uniq
        print(f"\n=== collision mass vs duplicate floor ({args.sid_index}, {time.time()-t0:.0f}s to load) ===")
        print(f"items in index:        {n_tok:,}")
        print(f"unique tuples:         {uniq:,}")
        print(f"collision mass (N-U):  {surplus:,}  ({surplus/max(n_tok,1):.4%})")

        g2 = df[df["digest"].isin(dup_groups.index)] if len(dup_groups) else df.iloc[0:0]
        all_same_sid, split_groups, missing = 0, 0, 0
        irreducible = 0
        biggest = []
        for dg, grp in g2.groupby("digest", sort=False):
            member_sids = sid.reindex(grp["item_id"].values)
            if member_sids.isna().any():
                missing += int(member_sids.isna().sum())
                member_sids = member_sids.dropna()
            codes = set(member_sids)
            if len(codes) == 0:
                continue
            if len(codes) == 1:
                all_same_sid += 1
                irreducible += len(grp) - 1
                if len(grp) >= 50:
                    biggest.append((int(len(grp)), codes.pop()))
            else:
                split_groups += 1
        print(f"dup groups fully in one SID bucket: {all_same_sid:,} / {len(dup_groups):,}")
        print(f"dup groups split across buckets:    {split_groups:,}  (identical rows, different codes: tie-break or index mismatch)")
        if missing:
            print(f"dup members missing from index:       {missing:,}")
        print(f"irreducible collision mass:           {irreducible:,}  "
              f"({irreducible/max(surplus,1):.2%} of observed collision mass)")
        print(f"=> reducible ceiling (any tokenizer): {surplus - irreducible:,}")
        if biggest:
            biggest.sort(reverse=True)
            print(f"largest same-SID duplicate groups (size, sid): {biggest[:args.top]}")
        report["sid_index"] = args.sid_index
        report.update({
            "index_n": n_tok, "index_unique_tuples": uniq,
            "collision_mass": surplus, "dup_groups_same_sid": all_same_sid,
            "dup_groups_split_sid": split_groups, "dup_members_missing": missing,
            "irreducible_mass": int(irreducible),
            "irreducible_share_of_mass": float(irreducible / max(surplus, 1)),
        })

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[scan] wrote {args.out}")


if __name__ == "__main__":
    main()
