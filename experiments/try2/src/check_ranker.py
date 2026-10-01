#!/usr/bin/env python3
"""check_ranker.py — Phase 1, Step 0: verify the ranker/tie hypothesis.

Phase 0 showed EXACT-name pairs capped at rank 80-180, which should be
impossible under a sane ranker (exact names score max jaccard). Hypothesis:
the old cheap score = max(jaccard, containment) saturates at 1.0 for any
subset-name candidate, flooding ties and burying the true pair by tie order.

This script, for a random sample of capped EXACT pairs:
  1. recomputes the OLD score (max(jaccard, containment)) for the pair and
     checks it against the stored score file  -> alignment sanity
  2. counts candidates in the same entity tied at the stored score
  3. recomputes BOTH orderings (old single-score vs new lexicographic
     jaccard-primary/containment-secondary from compute_cheap_scores_v2)
     and reports where the pair would rank under each

Verdicts:
  ALIGNMENT BUG     stored != recomputed -> fix before trusting ANY rank
  TIES CONFIRMED    most sampled pairs rank <80 under the new ordering
  NOT TIES          pairs stay deep even under the new ordering (true
                    ranking problem, look elsewhere)

Run BEFORE build_pool --rescore (needs the old score files).
~2-4 min.
"""

from __future__ import annotations

import os as _os, sys as _sys  # un-shadow stdlib 'inspect' (this dir has inspect.py)
_src_d = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0] or _os.getcwd()) == _src_d:
    _sys.path.pop(0)
    _sys.path.append(_src_d)

import argparse
import json
import time

import numpy as np

import normalize as N
from features import compute_cheap_scores_v2, _jac_contain

PRIMARY_K = 80


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    N.add_common_args(ap)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    t0 = time.time()
    cd = args.cache_dir
    meta = json.loads((cd / f"{args.split}_pool_meta.json").read_text())
    bin_bits = int(meta["bin_bits"])
    n_bins = int(meta["n_bins"])

    # ---- GT ----
    s1l, otl = [], []
    for s1, ids in N.stream_ground_truth(
            args.dataset / "train" / "train_ground_truth.tsv"):
        for c in ids:
            s1l.append(s1)
            otl.append(c)
    gt_s1 = np.asarray(s1l, np.int64)
    gt_ot = np.asarray(otl, np.int64)
    del s1l, otl
    p_gt = (gt_s1 << 33) | gt_ot
    order = np.argsort(p_gt, kind="stable")
    p_gt, gt_s1, gt_ot = p_gt[order], gt_s1[order], gt_ot[order]
    n_gt = len(p_gt)
    print(f"[rank0] GT pairs {n_gt:,}", flush=True)

    # ---- membership + rank ----
    found = np.zeros(n_gt, bool)
    rank = np.full(n_gt, 65535, np.uint32)
    for b in range(n_bins):
        nb = int(meta["bins"][b]["n"])
        if nb == 0:
            continue
        pp = np.memmap(cd / f"{args.split}_pool_bin{b:02d}.p64",
                       dtype=np.int64, mode="r", shape=(nb,))
        ra = np.fromfile(cd / f"{args.split}_pool_bin{b:02d}_rank.u16",
                         dtype=np.uint16)
        lo = int(np.searchsorted(p_gt, (b << bin_bits) << 33))
        hi = int(np.searchsorted(p_gt, ((b + 1) << bin_bits) << 33))
        if hi <= lo:
            continue
        gp = p_gt[lo:hi]
        pos = np.searchsorted(pp, gp)
        cp = np.clip(pos, 0, nb - 1)
        ok = (pos < nb) & (pp[cp] == gp)
        found[lo:hi] = ok
        rank[lo:hi] = np.where(ok, ra[cp].astype(np.uint32), 65535)
        del pp
    capped = np.flatnonzero(found & (rank >= PRIMARY_K))
    print(f"[rank0] capped pairs (rank>={PRIMARY_K}): {len(capped):,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- exact-name subset of capped pairs ----
    caches = {i: N.SourceCache(cd / f"{args.split}_source{i}")
              for i in (1, 2, 3)}
    m2 = gt_ot[capped] < N.S3_OFFSET
    r1 = caches[1].row_of(gt_s1[capped])
    ns1 = N.decode_str(caches[1].arrays["name_sorted"][np.maximum(r1, 0)])
    nso = np.zeros(len(capped), caches[1].arrays["name_sorted"].dtype)
    if m2.any():
        rr = caches[2].row_of(gt_ot[capped][m2])
        nso[m2] = caches[2].arrays["name_sorted"][np.maximum(rr, 0)]
    if (~m2).any():
        rr = caches[3].row_of(gt_ot[capped][~m2] - N.S3_OFFSET)
        nso[~m2] = caches[3].arrays["name_sorted"][np.maximum(rr, 0)]
    nso = N.decode_str(nso)
    exact_m = (ns1 == nso) & (ns1 != "")
    exact_idx = capped[exact_m]
    print(f"[rank0] capped EXACT pairs: {len(exact_idx):,}", flush=True)
    if len(exact_idx) == 0:
        print("[rank0] nothing to check — verdict: NOT APPLICABLE")
        return

    rng = np.random.default_rng(args.seed)
    take = rng.choice(exact_idx, size=min(args.sample, len(exact_idx)),
                      replace=False)

    # ---- per-sample analysis ----
    align_bad = 0
    n_checked = 0
    fixed_by_v2 = 0
    tie_counts = []
    entity_sizes = []
    old_ranks, new_ranks = [], []
    for gi in take:
        gi = int(gi)
        s1_id = int(gt_s1[gi])
        ot_id = int(gt_ot[gi])
        b = s1_id >> bin_bits
        nb = int(meta["bins"][b]["n"])
        if nb == 0:
            continue
        pp = np.memmap(cd / f"{args.split}_pool_bin{b:02d}.p64",
                       dtype=np.int64, mode="r", shape=(nb,))
        key = (np.int64(s1_id) << 33) | np.int64(ot_id)
        lo = int(np.searchsorted(pp, np.int64(s1_id) << 33))
        hi = int(np.searchsorted(pp, (np.int64(s1_id) + 1) << 33))
        if not (lo < hi and pp[np.clip(np.searchsorted(pp[lo:hi], key), 0, hi - lo - 1) + lo] == key):
            del pp
            continue
        ent_ot = np.asarray(pp[lo:hi]) & ((1 << 33) - 1)
        del pp
        pos_in = int(np.searchsorted(ent_ot, ot_id))
        if pos_in >= len(ent_ot) or int(ent_ot[pos_in]) != ot_id:
            continue
        sc = np.memmap(cd / f"{args.split}_pool_bin{b:02d}_score.f32",
                       dtype=np.float32, mode="r", shape=(nb,))[lo:hi]
        if len(sc) != len(ent_ot):
            continue
        # 1. old-score alignment: recompute max(jaccard, containment)
        s1_arr = np.full(1, s1_id, np.int64)
        ot_arr = ent_ot
        pri, sec = compute_cheap_scores_v2(caches[1], caches[2], caches[3],
                                            np.full(len(ent_ot), s1_id,
                                                    np.int64),
                                            ent_ot)
        # recompute native jaccard/containment for the old formula + alignment
        r1e = caches[1].row_of(s1_arr)
        # native-only old score for every candidate:
        from features import _gather_str, _other_rows, CHUNK  # noqa
        # (single entity, tiny — direct python)
        r1s = int(np.maximum(r1e[0], 0))
        a_str = N.decode_str(caches[1].arrays["name_sorted"][[r1s]])[0]
        old_fresh = np.zeros(len(ent_ot), np.float32)
        rows_o, is3 = _other_rows(ent_ot, caches[2], caches[3])
        for j in range(len(ent_ot)):
            rj = int(np.maximum(rows_o[j], 0))
            if is3[j]:
                b_str = N.decode_str(caches[3].arrays["name_sorted"][[rj]])[0]
            else:
                b_str = N.decode_str(caches[2].arrays["name_sorted"][[rj]])[0]
            if rows_o[j] < 0:
                b_str = ""
            jj, cc = _jac_contain(a_str, b_str)
            old_fresh[j] = max(jj, cc)
        if abs(float(old_fresh[pos_in]) - float(sc[pos_in])) > 1e-6:
            align_bad += 1
        # 2. ties at stored score
        tie_counts.append(int(np.sum(np.abs(sc - sc[pos_in]) <= 1e-9)))
        entity_sizes.append(len(ent_ot))
        # 3. old vs new ordering
        old_order = np.argsort(-sc, kind="stable")
        old_rank = int(np.flatnonzero(old_order == pos_in)[0])
        new_order = np.lexsort((-sec, -pri))
        new_rank = int(np.flatnonzero(new_order == pos_in)[0])
        old_ranks.append(old_rank)
        new_ranks.append(new_rank)
        if new_rank < PRIMARY_K:
            fixed_by_v2 += 1
        n_checked += 1

    if n_checked == 0:
        print("[rank0] no samples resolved — check pool files")
        return
    align_pct = align_bad / n_checked * 100
    fixed_pct = fixed_by_v2 / n_checked * 100
    print("\n=== STEP 0 RANKER CHECK (capped EXACT pairs) ===")
    print(f"  samples analysed:            {n_checked:,}")
    print(f"  stored-vs-recomputed mismatch: {align_bad} ({align_pct:.2f}%)")
    print(f"  median entity candidates:    {int(np.median(entity_sizes)):,}")
    print(f"  median ties at stored score: {int(np.median(tie_counts)):,}")
    print(f"  median OLD rank: {int(np.median(old_ranks))}   "
          f"median NEW (lex) rank: {int(np.median(new_ranks))}")
    print(f"  would rank <{PRIMARY_K} under new ordering: "
          f"{fixed_by_v2:,}/{n_checked:,} ({fixed_pct:.1f}%)")
    print()
    if align_pct > 1.0:
        print("VERDICT: ALIGNMENT BUG — stored scores don't match "
              "recomputation. Fix before trusting any rank numbers.")
    elif fixed_pct >= 70.0:
        print("VERDICT: TIES CONFIRMED — the containment-saturation tie "
              "flood is what buried these pairs; the lexicographic ranker "
              "fixes the majority. Proceed with Step 4 rescore.")
    else:
        print("VERDICT: NOT TIES — pairs stay deep even under the new "
              "ordering; ranking needs deeper investigation before Step 4.")
    print(f"\n[rank0] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
