#!/usr/bin/env python3
"""id_check.py — Phase 0, Step 1: rule out a trivial id-ordering solution.

Question: does an entity's ground-truth partner sit at a predictable POSITION
in its source file (i.e. the generator emitted records entity-by-entity, so
nearest-position = match)? If yes, blocking recall is a solved problem with a
one-line rule and the whole approach changes. We measure it before anything
else because it costs ~1 minute.

Three measures, per partner source (S2 and S3):
  1. correlation between normalized file positions of (S1 row, partner row)
  2. mean |position gap| vs a shuffled-pair baseline (same marginals)
  3. recall of the trivial "partner sits at my scaled position" rule
     (exact position and ±2 positions) vs the baseline hit rate

Verdict: CLEAN  -> positions carry no signal, proceed with normal blocking.
         RED FLAG -> report correlations and stop; discuss before Phase 1.

Stdlib + numpy only; streams the TSVs for ids (no cache needed).
"""

from __future__ import annotations

import os as _os, sys as _sys  # un-shadow stdlib 'inspect' (this dir has inspect.py)
_src_d = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0] or _os.getcwd()) == _src_d:
    _sys.path.pop(0)
    _sys.path.append(_src_d)

import argparse
import time
from pathlib import Path

import normalize as N


def stream_ids(path: Path) -> list:
    """Numeric parts of column-1 entity ids, in file order."""
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            tab = line.find("\t")
            tok = line if tab < 0 else line[:tab]
            tok = tok.strip()
            if not tok:
                continue
            try:
                _, n = N.parse_id(tok)
            except ValueError:
                continue
            out.append(n)
    return out


def positions_for(partner_ids: list, all_ids: list):
    """partner id -> its row position in its own source file."""
    import numpy as np
    arr = np.asarray(all_ids, np.int64)
    order = np.argsort(arr, kind="stable")
    sorted_ids = arr[order]
    p = np.searchsorted(sorted_ids, np.asarray(partner_ids, np.int64))
    p = np.clip(p, 0, len(sorted_ids) - 1)
    return order[p], sorted_ids[p], sorted_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    N.add_common_args(ap)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    gt_path = args.dataset / "train" / "train_ground_truth.tsv"
    if not gt_path.is_file():
        raise SystemExit(f"missing {gt_path}")
    print(f"[id_check] reading {gt_path}", flush=True)

    gt_s1_2, gt_s2 = [], []
    gt_s1_3, gt_s3 = [], []
    for s1, ids in N.stream_ground_truth(gt_path):
        for c in ids:
            if c < N.S3_OFFSET:
                gt_s1_2.append(s1)
                gt_s2.append(c)
            else:
                gt_s1_3.append(s1)
                gt_s3.append(c - N.S3_OFFSET)
    print(f"[id_check] GT pairs: s2={len(gt_s2):,}  s3={len(gt_s3):,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("[id_check] streaming source id order ...", flush=True)
    s1_ids = stream_ids(args.dataset / "train" / "train_source1.tsv")
    s2_ids = stream_ids(args.dataset / "train" / "train_source2.tsv")
    s3_ids = stream_ids(args.dataset / "train" / "train_source3.tsv")
    print(f"[id_check] rows: s1={len(s1_ids):,} s2={len(s2_ids):,} "
          f"s3={len(s3_ids):,} ({time.time()-t0:.0f}s)", flush=True)

    import numpy as np
    rng = np.random.default_rng(args.seed)

    def analyse(name: str, gt_s1: np.ndarray, partner_gt: np.ndarray,
                all_ids: list, n_all: int) -> dict:
        if len(partner_gt) == 0:
            return {}
        pos_partner, matched_sorted, _ = positions_for(partner_gt, all_ids)
        # position of the s1 row inside source1 (gt_s1 aligned)
        s1_sorted = np.sort(np.asarray(s1_ids, np.int64))
        q = np.searchsorted(s1_sorted, gt_s1)
        q = np.clip(q, 0, len(s1_sorted) - 1)
        pos_s1 = q  # gt ids come from source1 itself -> exact positions
        ok1 = s1_sorted[q] == gt_s1
        u = pos_s1.astype(np.float64) / max(n_all - 1, 1)
        n_src = len(all_ids)
        v = pos_partner.astype(np.float64) / max(n_src - 1, 1)
        okp = matched_sorted == partner_gt
        u, v = u[ok1 & okp], v[ok1 & okp]
        if len(u) < 30:
            print(f"[id_check]   {name}: too few aligned pairs ({len(u)})")
            return {}
        corr = float(np.corrcoef(u, v)[0, 1])
        gap = float(np.mean(np.abs(u - v)))
        # shuffled baseline (permute partner positions across pairs)
        vs = v[rng.permutation(len(v))]
        gap_shuf = float(np.mean(np.abs(u - vs)))
        # trivial rule: partner should sit at my scaled position
        pred = np.rint(u * (n_src - 1)).astype(np.int64)
        exact = float(np.mean(pred == np.rint(v * (n_src - 1)).astype(np.int64)))
        win = float(np.mean(
            np.abs(pred - np.rint(v * (n_src - 1)).astype(np.int64)) <= 2))
        win_shuf = float(np.mean(
            np.abs(pred - np.rint(vs * (n_src - 1)).astype(np.int64)) <= 2))
        r = {"n": len(u), "corr": corr, "gap": gap, "gap_shuf": gap_shuf,
             "exact": exact, "win5": win, "win5_shuf": win_shuf}
        print(f"\n[{name}]  n={r['n']:,}", flush=True)
        print(f"  position correlation (s1, partner): {corr:+.4f}")
        print(f"  mean |norm position gap|:           {gap:.4f}"
              f"   (shuffled baseline: {gap_shuf:.4f})")
        print(f"  trivial rule exact-position hit:    {exact*100:.4f}%")
        print(f"  trivial rule ±2-position hit:       {win*100:.4f}%"
              f"   (shuffled: {win_shuf*100:.4f}%)")
        return r

    r2 = analyse("vs source2", np.asarray(gt_s1_2, np.int64),
                 np.asarray(gt_s2, np.int64), s2_ids, len(s1_ids))
    r3 = analyse("vs source3", np.asarray(gt_s1_3, np.int64),
                 np.asarray(gt_s3, np.int64), s3_ids, len(s1_ids))

    red = False
    for r in (r2, r3):
        if not r:
            continue
        if abs(r["corr"]) > 0.2 or r["win5"] > 0.01 or r["win5_shuf"] > 0.001:
            red = True
    print("\n" + "=" * 70)
    if red:
        print("VERDICT: RED FLAG — file position correlates with GT.")
        print("         Do NOT proceed; bring this report back first.")
    else:
        print("VERDICT: CLEAN — id/file position carries no usable signal.")
        print("         A trivial nearest-position rule is not the task.")
    print("=" * 70)
    print(f"[id_check] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
