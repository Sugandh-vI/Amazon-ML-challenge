#!/usr/bin/env python3
"""cap_plan.py — Phase 1, Step 5a: choose per-family per-key caps.

Reads the tracer's per-victim (min bucket size, best family) and the exact
per-family shared-key pair-count arrays, then greedily raises family caps
one level at a time, always taking the best (victims recovered) per
(added volume) step, until the volume budget is exhausted.

Writes {split}_cap_plan.json which count_families and build_pool read
automatically — family caps apply from then on (stage A is invalidated via
plan_sig, so no stale pool can survive a plan change).

Budget default = 540M = the 600M FULL-materialization threshold minus 10%
headroom for the two new KEY_VERSION-2 families (hnstate, atok2).

Requires a tracer run made with the current trace_misses.py (npz carries
capv_fam). Runtime: seconds to ~2 min.
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
from pathlib import Path

import numpy as np

import normalize as N
from blocking import FAMILIES, KEY_VERSION
from cap_curve import family_pc

LEVELS = [7_500, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    N.add_common_args(ap)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--budget", type=int, default=540_000_000,
                    help="max post-cap volume for the plan "
                         "(default 540M = 600M threshold - 10%% headroom "
                         "for the new key families)")
    ap.add_argument("--levels", default="",
                    help="comma-separated cap levels (default: built-ins)")
    args = ap.parse_args()

    t0 = time.time()
    levels = ([int(x) for x in args.levels.split(",") if x.strip()]
              if args.levels else LEVELS)
    cd = args.cache_dir
    npz_p = cd / f"{args.split}_trace.npz"
    summ_p = cd / f"{args.split}_trace_summary.json"
    if not npz_p.is_file():
        raise SystemExit("missing tracer output — run trace_misses first")
    tr = np.load(npz_p)
    if "capv_fam" not in tr.files:
        raise SystemExit(
            "trace npz has no capv_fam — re-run the UPDATED trace_misses.py "
            "(it stamps each victim with its best family)")
    minpc = tr["capv_minpc"].astype(np.int64)
    vfam = tr["capv_fam"].astype(np.int64)
    summ = json.loads(summ_p.read_text())
    n_gt = int(summ["n_gt"])
    n_cap = int(summ["reasons"].get("CAP", 0))
    n_nokey = int(summ["reasons"].get("NOKEY", 0))
    n_bug = int(summ["reasons"].get("BUG", 0))

    # ---- per-family volume curves (exact, from key counts) ----
    print("[plan] loading per-family pc arrays ...", flush=True)
    fpcs = [family_pc(fi, cd, args.split) for fi in range(len(FAMILIES))]
    vols = {}
    for fi, fam in enumerate(FAMILIES):
        for L in [5_000] + levels:
            v = 0
            for pc in fpcs[fi]:
                if len(pc):
                    v += int(pc[pc <= L].sum())
            vols[(fi, L)] = v
    base_vol = sum(vols[(fi, 5_000)] for fi in range(len(FAMILIES)))
    print(f"[plan] base volume @5000 = {base_vol:,} "
          f"(counts post_cap should match); budget = {args.budget:,}",
          flush=True)

    caps = {fi: 5_000 for fi in range(len(FAMILIES))}
    spent = base_vol
    fam_recovered = {fi: 0 for fi in range(len(FAMILIES))}

    def delta_r(fi: int, old_c: int, new_c: int) -> int:
        m = (vfam == fi) & (minpc > old_c) & (minpc <= new_c)
        return int(m.sum())

    print("[plan] greedy raise ...", flush=True)
    while spent < args.budget:
        best = None
        for fi, fam in enumerate(FAMILIES):
            cur = caps[fi]
            for L in levels:
                if L <= cur:
                    continue
                dv = vols[(fi, L)] - vols[(fi, cur)]
                if dv <= 0:
                    continue
                dr = delta_r(fi, cur, L)
                if dr <= 0:
                    continue
                if spent + dv > args.budget:
                    continue
                ratio = dr / dv
                if best is None or ratio > best[0]:
                    best = (ratio, fi, L, dv, dr)
        if best is None:
            break
        _, fi, L, dv, dr = best
        caps[fi] = L
        spent += dv
        fam_recovered[fi] += dr
    recovered = int((minpc <= np.array([caps[f] for f in vfam])).sum())

    # ---- report + plan file ----
    proj_covered = n_gt - n_nokey - n_bug - (n_cap - recovered)
    lines = []

    def L_(s=""):
        lines.append(s)
        print(s, flush=True)

    L_("=" * 76)
    L_(f"CAP PLAN — split={args.split} — budget {args.budget:,}")
    L_("=" * 76)
    L_(f"  {'family':8s} {'cap':>9s} {'added volume':>14s} "
      f"{'victims recovered':>18s}")
    for fi, fam in enumerate(FAMILIES):
        added = vols[(fi, caps[fi])] - vols[(fi, 5_000)]
        L_(f"  {fam:8s} {caps[fi]:>9,} {added:>14,} "
           f"{fam_recovered[fi]:>18,}")
    L_("-" * 76)
    L_(f"  projected post-cap volume: {spent:,} "
       f"({'FULL' if spent <= 600_000_000 else 'TRIM'} at the 600M rule)")
    L_(f"  CAP victims recovered:     {recovered:,} / {n_cap:,}")
    L_(f"  residue (keys target):     {n_cap - recovered:,} "
       f"-> hnstate/atok2 + next cycle")
    today = (n_gt - n_nokey - n_bug - n_cap) / max(n_gt, 1) * 100
    L_(f"  projected pair coverage:   {proj_covered:,}/{n_gt:,} "
       f"({proj_covered / max(n_gt, 1) * 100:.2f}% vs {today:.2f}% today)")
    L_(f"  elapsed: {time.time()-t0:.0f}s")

    plan = {"split": args.split, "key_version": KEY_VERSION,
            "budget": args.budget, "per_family_cap": {FAMILIES[fi]: caps[fi]
                                                      for fi in range(len(FAMILIES))},
            "projected_volume": int(spent),
            "recovered_cap_victims": int(recovered),
            "residue_cap_victims": int(n_cap - recovered),
            "projected_pair_coverage": round(proj_covered / max(n_gt, 1), 6),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (cd / f"{args.split}_cap_plan.json").write_text(
        json.dumps(plan, indent=1))
    out = N.TRY2_DIR / "output"
    out.mkdir(exist_ok=True)
    (out / "cap_plan.txt").write_text("\n".join(lines) + "\n")
    print(f"\n[plan] wrote {args.split}_cap_plan.json — next: "
          f"python src/count_families.py --split {args.split} "
          f"(auto-rebuilds keys, applies plan)")


if __name__ == "__main__":
    main()
