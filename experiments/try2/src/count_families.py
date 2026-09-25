#!/usr/bin/env python3
"""count_families.py — Phase 0, Step 2: true uncapped family sizes.

Count-only pass over every blocking family: streams each cache once per family
(bounded RAM: one family-side key array at a time), computes per-side key
frequencies, and derives EXACT pair counts under three regimes:

  uncapped    sum over shared keys of c1*c2            (what the keys really produce)
  post_cap    buckets with pairs <= per_key_cap        (try1's cap semantics, unchanged)
  budget_kept what try1's per-side budget actually kept (ascending-size keep order)

No pair materialization happens here. As a side effect every (family, side)
key+entity array is saved to disk — this IS the Phase-1 key cache, so later
re-runs with different caps/budgets skip the streaming pass entirely.

The printed table + {split}_blocking_counts.json are the inputs build_pool.py
uses to apply the approved decision rule (full materialize vs informed trim).

Run before build_pool. ~6-10 min on an M4 with --workers 4.
"""

from __future__ import annotations

import os as _os, sys as _sys  # un-shadow stdlib 'inspect' (this dir has inspect.py)
_src_d = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0] or _os.getcwd()) == _src_d:
    _sys.path.pop(0)
    _sys.path.append(_src_d)

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

import normalize as N
from blocking import FAMILIES, build_all_keys


def _counts_after_selection(pc: np.ndarray, cap: int, budget: int) -> dict:
    """Replicates expand_join's cap filter + budget keep, returns pair counts."""
    keep_mask = pc <= cap
    cap_dropped_pairs = int(pc[~keep_mask].sum())
    cap_dropped_buckets = int((~keep_mask).sum())
    pc_ok = pc[keep_mask]
    post_cap = int(pc_ok.sum())
    if len(pc_ok) == 0:
        return {"uncapped": int(pc.sum()), "post_cap": 0,
                "budget_kept": 0, "budget_dropped_pairs": 0,
                "budget_dropped_buckets": 0, "cap_dropped_pairs": cap_dropped_pairs,
                "cap_dropped_buckets": cap_dropped_buckets,
                "buckets": int(len(pc)), "buckets_post_cap": 0}
    order = np.argsort(pc_ok, kind="stable")
    cum = np.cumsum(pc_ok[order])
    keep = int(np.searchsorted(cum, budget, side="right")) + 1
    keep = min(keep, len(order))
    budget_kept = int(pc_ok[order[:keep]].sum())
    return {
        "uncapped": int(pc.sum()),
        "post_cap": post_cap,
        "budget_kept": budget_kept,
        "budget_dropped_pairs": post_cap - budget_kept,
        "budget_dropped_buckets": int(len(order) - keep),
        "cap_dropped_pairs": cap_dropped_pairs,
        "cap_dropped_buckets": cap_dropped_buckets,
        "buckets": int(len(pc)),
        "buckets_post_cap": int(len(pc_ok)),
    }


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """np.save to a temp name, then rename into place (np.save adds .npy)."""
    tmp = path.parent / (path.name[:-4] + "_.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def count_one_family(fi: int, split: str, dataset: str, cache_dir: str,
                     per_key_cap: int, side_budget: int,
                     limit: Optional[int]) -> dict:
    """Worker: build keys for one family (3 sides), count, cache keys to disk."""
    t0 = time.time()
    cd = Path(cache_dir)
    kdir = cd / "pool_keys"
    kdir.mkdir(parents=True, exist_ok=True)
    fam = FAMILIES[fi]
    sides = {}
    rows = {}
    for side in (1, 2, 3):
        cache = N.SourceCache(cd / f"{split}_source{side}")
        rows[side] = cache.rows
        k, e, _ = build_all_keys(cache, s3=(side == 3), only=fi)
        _atomic_save(kdir / f"{split}_f{fi}_s{side}_keys.npy", k)
        _atomic_save(kdir / f"{split}_f{fi}_s{side}_ents.npy", e)
        u, c = np.unique(k, return_counts=True)
        sides[side] = (u, c)
        del k, e
    res = {"family": fam, "fi": fi, "sides": {}}
    for other in (2, 3):
        u1, c1 = sides[1]
        uo, co = sides[other]
        _, i1, io = np.intersect1d(u1, uo, return_indices=True)
        pc = c1[i1].astype(np.int64) * co[io]
        res["sides"][f"s{other}"] = _counts_after_selection(
            pc, per_key_cap, side_budget)
    res["keys"] = {f"s{s}": int(len(sides[s][0])) for s in (1, 2, 3)}
    res["rows_used"] = {f"s{s}": int(rows[s]) for s in (1, 2, 3)}
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    N.add_common_args(ap)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel family workers (default: min(4, cpu_count))")
    ap.add_argument("--per-key-cap", type=int, default=5000,
                    help="try1's per-bucket pair cap (unchanged in Phase 0)")
    ap.add_argument("--force", action="store_true", help="redo even if counts exist")
    args = ap.parse_args()

    workers = args.workers or min(4, _os.cpu_count() or 1)
    # try1 budget: total 200M split across 8 families x 2 sides
    side_budget = 200_000_000 // (len(FAMILIES) * 2)
    json_path = args.cache_dir / f"{args.split}_blocking_counts.json"

    # make sure caches exist (parent builds them once; workers only read)
    N.resolve_split_stems(args.split, args.dataset, args.cache_dir,
                          limit=args.limit, force=False)

    key_paths = [args.cache_dir / "pool_keys" /
                 f"{args.split}_f{fi}_s{s}_keys.npy"
                 for fi in range(len(FAMILIES)) for s in (1, 2, 3)]
    if json_path.is_file() and all(p.is_file() for p in key_paths) and not args.force:
        print(f"[count] {json_path.name} + all key files exist — skipping "
              f"(use --force to redo)")
        data = json.loads(json_path.read_text())
        _print_table(data)
        return

    print(f"[count] families={len(FAMILIES)} workers={workers} "
          f"per_key_cap={args.per_key_cap} try1_side_budget={side_budget:,}",
          flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(count_one_family, fi, args.split, str(args.dataset),
                          str(args.cache_dir), args.per_key_cap, side_budget,
                          args.limit): fi for fi in range(len(FAMILIES))}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[count]   {r['family']:6s} done ({r['elapsed_s']}s) "
                  f"keys=({r['keys']['s1']:,}|{r['keys']['s2']:,}|"
                  f"{r['keys']['s3']:,})", flush=True)
    results.sort(key=lambda r: r["fi"])

    families = {}
    totals = {"uncapped": 0, "post_cap": 0, "budget_kept": 0,
              "cap_dropped_pairs": 0, "cap_dropped_buckets": 0,
              "budget_dropped_pairs": 0}
    for r in results:
        fam = r["family"]
        s2, s3 = r["sides"]["s2"], r["sides"]["s3"]
        entry = {
            "keys": r["keys"],
            "s2": s2, "s3": s3,
            "uncapped": s2["uncapped"] + s3["uncapped"],
            "post_cap": s2["post_cap"] + s3["post_cap"],
            "budget_kept": s2["budget_kept"] + s3["budget_kept"],
            "cap_dropped_pairs": s2["cap_dropped_pairs"] + s3["cap_dropped_pairs"],
            "elapsed_s": r["elapsed_s"],
        }
        families[fam] = entry
        for k in totals:
            totals[k] += int(s2.get(k, 0)) + int(s3.get(k, 0))

    data = {
        "split": args.split,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_key_cap": args.per_key_cap,
        "try1_side_budget": side_budget,
        "families": families,
        "totals": totals,
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(json_path)
    _print_table(data)
    print(f"[count] wrote {json_path}")
    print(f"[count] done in {data['elapsed_s']}s")


def _print_table(data: dict) -> None:
    print("\n=== family size counts (pairs) ===")
    hdr = (f"{'family':7s} {'uncapped':>13s} {'post-cap':>13s} "
           f"{'try1-kept':>13s} {'cap-dropped':>13s} {'keys(s1/s2/s3)':>22s}")
    print(hdr)
    print("-" * len(hdr))
    for fam, e in data["families"].items():
        print(f"{fam:7s} {e['uncapped']:>13,} {e['post_cap']:>13,} "
              f"{e['budget_kept']:>13,} {e['cap_dropped_pairs']:>13,} "
              f"{e['keys']['s1']:>6,}/{e['keys']['s2']:>6,}/{e['keys']['s3']:>6,}")
    t = data["totals"]
    print("-" * len(hdr))
    print(f"{'TOTAL':7s} {t['uncapped']:>13,} {t['post_cap']:>13,} "
          f"{t['budget_kept']:>13,} {t['cap_dropped_pairs']:>13,}")
    print(f"\ntotals: uncapped={t['uncapped']:,}  "
          f"post-cap(materializable)={t['post_cap']:,}  "
          f"try1-budget-kept={t['budget_kept']:,}")
    print(f"try1's budget hid {t['post_cap'] - t['budget_kept']:,} post-cap pairs; "
          f"the per-key cap removes {t['cap_dropped_pairs']:,} "
          f"({t['cap_dropped_buckets']:,} buckets) — cap UNCHANGED in Phase 0.")
    cap_full, cap_target = 600_000_000, 500_000_000
    if t["post_cap"] <= cap_full:
        print(f"[decision preview] post-cap {t['post_cap']:,} <= {cap_full:,} "
              f"-> build_pool will run FULL materialization.")
    else:
        print(f"[decision preview] post-cap {t['post_cap']:,} > {cap_full:,} "
              f"-> build_pool will TRIM to {cap_target:,} (informed, logged).")


if __name__ == "__main__":
    main()
