#!/usr/bin/env python3
"""gate_check.py — Phase 1, Step 6: the standing gate, measured.

GATE: stage-A entity recall >= 98% on train GT, where an entity counts as
recalled iff EVERY one of its GT pairs is present in the pool at ANY rank.
(Model/rank work is forbidden until this passes.)

Prints:
  pair in-pool % (any rank)          — never-gen leakage shows here
  entity all-in-pool %  <- GATE      PASS/FAIL at 98%
  pair recall @K / entity all-found @K for K in --ks
  residual breakdown vs last tracer (if trace_summary exists)

Standalone (membership + rank scan): ~60-90 s. Run after every build.
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

GATE = 0.98


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    N.add_common_args(ap)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--ks", default="80,150,300,500")
    args = ap.parse_args()

    t0 = time.time()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    cd = args.cache_dir
    meta = json.loads((cd / f"{args.split}_pool_meta.json").read_text())
    bin_bits = int(meta["bin_bits"])
    n_bins = int(meta["n_bins"])

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
    found = np.zeros(n_gt, bool)
    rank = np.full(n_gt, 65535, np.uint32)
    for b in range(n_bins):
        nb = int(meta["bins"][b]["n"])
        if nb == 0:
            continue
        pp = np.memmap(cd / f"{args.split}_pool_bin{b:02d}.p64",
                       dtype=np.int64, mode="r", shape=(nb,))
        lo = int(np.searchsorted(p_gt, (b << bin_bits) << 33))
        hi = int(np.searchsorted(p_gt, ((b + 1) << bin_bits) << 33))
        if hi <= lo:
            continue
        gp = p_gt[lo:hi]
        pos = np.searchsorted(pp, gp)
        cp = np.clip(pos, 0, nb - 1)
        ok = (pos < nb) & (pp[cp] == gp)
        found[lo:hi] = ok
        rp = np.fromfile(cd / f"{args.split}_pool_bin{b:02d}_rank.u16",
                         dtype=np.uint16)
        rank[lo:hi] = np.where(ok, rp[cp].astype(np.uint32), 65535)
        del pp
    print(f"[gate] scanned {n_gt:,} GT pairs ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- entity aggregation (vectorized) ----
    ents, gt_cnt = np.unique(gt_s1, return_counts=True)

    def entity_rate(mask: np.ndarray) -> float:
        ents_m, cm = np.unique(gt_s1[mask], return_counts=True)
        pos = np.searchsorted(ents, ents_m)
        hit = np.zeros(len(ents), np.int64)
        hit[pos] = cm
        return float((hit == gt_cnt).mean())

    pair_in = float(found.mean())
    ent_in = entity_rate(found)

    print()
    print("=" * 64)
    print(f"STAGE-A GATE CHECK — split={args.split} — pool {meta.get('pool_total', 0):,} pairs")
    print("=" * 64)
    print(f"  pair in-pool (any rank):        {pair_in * 100:.2f}%")
    print(f"  ENTITY all-in-pool (GATE 98%):  {ent_in * 100:.2f}%   "
          f"-> {'PASS' if ent_in >= GATE else 'FAIL'}")
    print()
    print(f"  {'K':>6s} {'pair-recall':>13s} {'entity all-found':>18s}")
    for k in ks:
        m = found & (rank < k)
        pr = float(m.mean())
        er = entity_rate(m)
        print(f"  {k:>6} {pr * 100:>12.2f}% {er * 100:>17.2f}%")
    summ_p = cd / f"{args.split}_trace_summary.json"
    if summ_p.is_file():
        s = json.loads(summ_p.read_text())
        print(f"\n  last tracer: {json.dumps(s.get('reasons', {}))} "
              f"(re-run trace_misses for a fresh split after this build)")
    verdict = ("PASS — Phase 2 unlocked (still verify rank @K before "
               "training)" if ent_in >= GATE
               else "FAIL — stay in Phase 1 (next cycle: caps/keys)")
    print(f"\n[gate] {verdict}")
    print(f"[gate] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
