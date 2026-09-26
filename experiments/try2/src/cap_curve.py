#!/usr/bin/env python3
"""cap_curve.py — Phase 1, Step 2: cap volume/coverage tradeoff.

Two exact measurements, no pool rebuild:

  VOLUME   for each cap in --caps, the post-cap stage-A pair volume
           (what build_pool would materialize), computed from the cached
           per-key count arrays — seconds per cap.
  COVERAGE the GT pair recall of stage-A at that cap, computed from the
           tracer artifacts: pairs already in the pool (cap=5000) plus
           CAP-victim misses whose minimum bucket size <= cap, minus BUG
           and NOKEY misses (unaffected by the cap).

Prints one table: cap | stage-A volume | over 600M threshold? | GT coverage.
Writes output/cap_curve.txt.

Run AFTER trace_misses. ~4-6 min (mostly the first run building the
per-family intersection stats, which it shares with trace_misses' cache).
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
from blocking import FAMILIES


def family_pc(fi: int, cache_dir: Path, split: str):
    """All shared-key pair counts for one family (S1 x S2 and S1 x S3).

    Prefers trace_misses' precomputed unique/count arrays (exact same data,
    no resort).
    """
    tdir = cache_dir / "trace_keys"
    kdir = cache_dir / "pool_keys"
    # family not in this pool's key generation (old pool, new scripts):
    # contribute nothing instead of crashing on a missing file.
    if not all((kdir / f"{split}_f{fi}_s{s}_keys.npy").is_file()
               for s in (1, 2, 3)):
        return [np.empty(0, np.int64), np.empty(0, np.int64)]

    def uc(s: int):
        up = tdir / f"{split}_f{fi}_s{s}_u.npy"
        cp = tdir / f"{split}_f{fi}_s{s}_c.npy"
        if up.is_file() and cp.is_file():
            return (np.load(up, mmap_mode="r"), np.load(cp, mmap_mode="r"))
        k = np.load(kdir / f"{split}_f{fi}_s{s}_keys.npy", mmap_mode="r")
        return np.unique(k, return_counts=True)

    u1, c1 = uc(1)
    out = []
    for s in (2, 3):
        uo, co = uc(s)
        _, i1, io = np.intersect1d(u1, uo, return_indices=True)
        pc = c1[i1].astype(np.int64) * co[io]
        out.append(pc)
        del uo, co
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    N.add_common_args(ap)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--caps", default="5000,20000,50000",
                    help="comma-separated per-key caps to evaluate")
    ap.add_argument("--max-volume", type=int, default=600_000_000,
                    help="full-materialize threshold from the Phase-0 rule")
    args = ap.parse_args()

    t0 = time.time()
    caps = [int(x) for x in args.caps.split(",") if x.strip()]
    cd = args.cache_dir
    trace_p = cd / f"{args.split}_trace.npz"
    summ_p = cd / f"{args.split}_trace_summary.json"
    if not (trace_p.is_file() and summ_p.is_file()):
        raise SystemExit("missing tracer outputs — run trace_misses first")
    summ = json.loads(summ_p.read_text())
    tr = np.load(trace_p)
    capv_pc = tr["capv_minpc"]
    n_gt = int(summ["n_gt"])
    n_nokey = int(summ["reasons"].get("NOKEY", 0))
    n_bug = int(summ["reasons"].get("BUG", 0))
    n_cap = int(summ["reasons"].get("CAP", 0))
    # stage-A coverage at cap 5000 = GT - all misses
    base_covered = n_gt - n_nokey - n_bug - n_cap
    print(f"[cap] GT={n_gt:,}  in-pool@5000={base_covered:,}  "
          f"misses: cap={n_cap:,} nokey={n_nokey:,} bug={n_bug:,}",
          flush=True)

    # ---- volumes ----
    print("[cap] computing per-family shared-key pc arrays ...", flush=True)
    per_family_pc = [family_pc(fi, cd, args.split)
                     for fi in range(len(FAMILIES))]

    lines = []

    def L(s=""):
        lines.append(s)
        print(s, flush=True)

    L("=" * 78)
    L(f"CAP CURVE — split={args.split} — coverage = GT pair recall of stage-A")
    L("=" * 78)
    L(f"  {'cap':>8s} {'stage-A volume':>16s} {'over 600M?':>11s} "
      f"{'GT coverage':>13s} {'vs cap5000':>12s} {'cap-victims saved':>18s}")
    prev_cov = None
    for cap in caps:
        vol = 0
        for fi in range(len(FAMILIES)):
            for pc in per_family_pc[fi]:
                vol += int(pc[pc <= cap].sum()) if len(pc) else 0
        saved = int((capv_pc <= cap).sum()) if len(capv_pc) else 0
        cov = (base_covered + saved) / max(n_gt, 1) * 100
        over = "YES" if vol > args.max_volume else "no"
        delta = "" if prev_cov is None else f"{cov - prev_cov:+.2f}pp"
        L(f"  {cap:>8,} {vol:>16,} {over:>11} {cov:>12.2f}% "
          f"{delta:>12} {saved:>18,}")
        if prev_cov is None:
            prev_cov = cov
    # current-truth row
    saved5 = int((capv_pc <= 5000).sum()) if len(capv_pc) else 0
    L()
    L(f"  note: BUG ({n_bug:,}) + NOKEY ({n_nokey:,}) misses are "
      f"cap-independent — fixing those is Steps 3/5, not the cap.")
    L(f"  cap=5000 sanity: saved-at-5000 should be 0 -> {saved5:,}")
    L(f"  elapsed: {time.time()-t0:.0f}s")

    out_dir = N.TRY2_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "cap_curve.txt").write_text("\n".join(lines) + "\n")
    print(f"\n[cap] wrote {out_dir / 'cap_curve.txt'}")


if __name__ == "__main__":
    main()
