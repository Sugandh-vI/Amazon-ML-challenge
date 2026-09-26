#!/usr/bin/env python3
"""build_pool.py — Phase 0, Step 3: materialize + annotate the candidate pool.

Reads {split}_blocking_counts.json (from count_families) and applies the
approved decision rule:

  post-cap volume <= --max-volume  -> FULL materialization (budget = infinite)
  post-cap volume >  --max-volume  -> INFORMED TRIM to --target-volume: a
      uniform scale factor is applied to every family/side join budget
      (equal trim fraction across families — no key-level preferences), and
      the exact fractions are printed and stored in pool_meta.json.

Stages (each resumable, each logged):
  A. per-family join using expand_join_stream — selection semantics are copied
     VERBATIM from blocking.expand_join (same cap filter, same ascending-size
     budget keep); only the emission is chunked to bound RAM. Output appended
     to {split}_stageA_fam{fi}.p64 (packed int64 pairs).
  B. scatter family files into s1-range bins ({split}_pool_binNN.p64).
  C. per-bin np.unique (global dedup across families).
  D. per-bin cheap-score + per-entity score-desc RANK (parallel workers),
     written as {bin}_score.f32 + {bin}_rank.u16.

Result: an annotated pool where every candidate knows its entity's score rank,
which analyze_misses consumes for decomposition + recall@K — one pass, many K.

--force rebuilds pool artifacts (the key cache from count_families is kept).
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

import numpy as np

import normalize as N
from blocking import FAMILIES, KEY_VERSION, pack_pairs
from features import compute_cheap_scores_v2

BIN_BITS = 17                    # s1 bins of 131072 ids
BIG = 1 << 60                    # "infinite" budget for full materialization


# ---------------------------------------------------------------- stage A
def expand_join_stream(k1, e1, k2, e2, per_key_cap, budget, emit,
                       chunk=4_000_000):
    """blocking.expand_join selection, chunked emission via emit(s1, ot).

    Selection logic (cap filter + ascending-size budget keep + bucket order)
    is copied verbatim from try1's expand_join so results are identical.
    Returns (dropped_cap_buckets, dropped_buckets_budget, n_buckets, total).
    """
    o1 = np.argsort(k1, kind="stable")
    s1k, s1e = k1[o1], e1[o1]
    o2 = np.argsort(k2, kind="stable")
    s2k, s2e = k2[o2], e2[o2]
    u1, st1, c1 = np.unique(s1k, return_index=True, return_counts=True)
    u2, st2, c2 = np.unique(s2k, return_index=True, return_counts=True)
    common, i1, i2 = np.intersect1d(u1, u2, return_indices=True)
    if len(common) == 0:
        return 0, 0, 0, 0
    pc = c1[i1].astype(np.int64) * c2[i2]
    ok = pc <= per_key_cap
    dropped_cap = int((~ok).sum())
    common, i1, i2, pc = common[ok], i1[ok], i2[ok], pc[ok]
    dropped_budget = 0
    if len(pc) and int(pc.sum()) > budget:
        order = np.argsort(pc, kind="stable")
        cum = np.cumsum(pc[order])
        keep = int(np.searchsorted(cum, budget, side="right")) + 1
        keep = min(keep, len(order))
        sel = np.sort(order[:keep])
        dropped_budget = len(order) - keep
        common, i1, i2, pc = common[sel], i1[sel], i2[sel], pc[sel]
    total = int(pc.sum()) if len(pc) else 0
    acc1, acc2, acc_n = [], [], 0
    for b in range(len(common)):
        a_st, a_n = int(st1[i1[b]]), int(c1[i1[b]])
        b_st, b_n = int(st2[i2[b]]), int(c2[i2[b]])
        seg1 = np.repeat(s1e[a_st:a_st + a_n], b_n)
        seg2 = np.tile(s2e[b_st:b_st + b_n], a_n)
        acc1.append(seg1)
        acc2.append(seg2)
        acc_n += a_n * b_n
        if acc_n >= chunk:
            emit(np.concatenate(acc1), np.concatenate(acc2))
            acc1.clear()
            acc2.clear()
            acc_n = 0
    if acc_n:
        emit(np.concatenate(acc1), np.concatenate(acc2))
    return dropped_cap, dropped_budget, len(common), total


def materialize_family(fi: int, split: str, cache_dir: str,
                       budgets: dict, per_key_cap: int, force: bool,
                       plan_sig: str = "") -> dict:
    """Stage-A worker: join S1 x {S2,S3} for one family, append packed pairs."""
    cd = Path(cache_dir)
    kdir = cd / "pool_keys"
    out = cd / f"{split}_stageA_fam{fi}.p64"
    done = cd / f"{split}_stageA_fam{fi}.done"
    if out.is_file() and done.is_file() and not force:
        meta = json.loads(done.read_text())
        if (out.stat().st_size == meta.get("bytes", -1)
                and meta.get("plan_sig") == plan_sig):
            return {"family": FAMILIES[fi], "skipped": True, **meta}
    t0 = time.time()
    k1 = np.load(kdir / f"{split}_f{fi}_s1_keys.npy", mmap_mode="r")
    e1 = np.load(kdir / f"{split}_f{fi}_s1_ents.npy", mmap_mode="r")
    tmp = cd / f"{split}_stageA_fam{fi}.p64.tmp"
    fh = open(tmp, "wb")

    def emit(s1_chunk, ot_chunk):
        packed = pack_pairs(s1_chunk, ot_chunk)
        fh.write(packed.astype("<i8").tobytes())

    stats = {"cap": 0, "bud": 0, "buckets": 0, "total": 0, "s2": 0, "s3": 0}
    for side in (2, 3):
        k2 = np.load(kdir / f"{split}_f{fi}_s{side}_keys.npy", mmap_mode="r")
        e2 = np.load(kdir / f"{split}_f{fi}_s{side}_ents.npy", mmap_mode="r")
        dc, db, nb, tot = expand_join_stream(
            k1, e1, k2, e2, per_key_cap, budgets[f"s{side}"], emit)
        stats["cap"] += dc
        stats["bud"] += db
        stats["buckets"] += nb
        stats["total"] += tot
        stats[f"s{side}"] = tot
        del k2, e2
    fh.close()
    os.replace(tmp, out)
    meta = {"family": FAMILIES[fi], "bytes": out.stat().st_size,
            "plan_sig": plan_sig,
            "total_pairs": stats["total"], "cap_dropped_buckets": stats["cap"],
            "budget_dropped_buckets": stats["bud"],
            "s2_pairs": stats["s2"], "s3_pairs": stats["s3"],
            "elapsed_s": round(time.time() - t0, 1)}
    done.write_text(json.dumps(meta))
    return {"family": FAMILIES[fi], "skipped": False, **meta}


# ---------------------------------------------------------------- stage B
def scatter_stage_a(split: str, cache_dir: Path, n_bins: int, force: bool) -> None:
    marker = cache_dir / f"{split}_pool_scatter.done"
    fam_sizes = [cache_dir / f"{split}_stageA_fam{fi}.p64" for fi in
                 range(len(FAMILIES))]
    sig = [p.stat().st_size if p.is_file() else -1 for p in fam_sizes]
    if marker.is_file() and not force:
        m = json.loads(marker.read_text())
        if m.get("fam_sizes") == sig:
            print("[pool] scatter: up to date", flush=True)
            return
    bins = [cache_dir / f"{split}_pool_bin{b:02d}.p64" for b in range(n_bins)]
    fh = [open(b, "wb") for b in bins]
    t0 = time.time()
    total = 0
    for fi in range(len(FAMILIES)):
        p = cache_dir / f"{split}_stageA_fam{fi}.p64"
        n = p.stat().st_size // 8
        m = np.memmap(p, dtype=np.int64, mode="r", shape=(n,))
        for lo in range(0, n, 8_000_000):
            seg = m[lo:lo + 8_000_000]
            b = (seg >> 50).astype(np.int64)   # (s1 >> 17) with s1 = seg >> 33
            if b.max(initial=0) >= n_bins:
                raise SystemExit(f"[pool] s1 id beyond s1_max — rerun with "
                                 f"fresh caches (bin table stale)")
            for bb in np.unique(b):
                fh[int(bb)].write(seg[b == bb].astype("<i8").tobytes())
        total += n
        del m
        print(f"[pool]   scattered {FAMILIES[fi]:6s} ({n:,} pairs)", flush=True)
    for f in fh:
        f.close()
    marker.write_text(json.dumps({"fam_sizes": sig, "n_bins": n_bins,
                                  "total": total,
                                  "elapsed_s": round(time.time() - t0, 1)}))
    print(f"[pool] scatter done: {total:,} pairs -> {n_bins} bin(s) "
          f"({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- stage C
def dedup_bins(split: str, cache_dir: Path, n_bins: int, force: bool) -> list:
    marker = cache_dir / f"{split}_pool_dedup.done"
    sizes = {}
    if marker.is_file() and not force:
        sizes = json.loads(marker.read_text())
    out = []
    for b in range(n_bins):
        p = cache_dir / f"{split}_pool_bin{b:02d}.p64"
        if str(b) in sizes and p.is_file() and not force:
            n = sizes[str(b)]
            if p.stat().st_size == n * 8:
                out.append(n)
                continue
        n = p.stat().st_size // 8
        m = np.memmap(p, dtype=np.int64, mode="r", shape=(n,))
        u = np.unique(m)
        del m
        tmp = cache_dir / f"{split}_pool_bin{b:02d}_.p64"
        u.astype("<i8").tofile(tmp)
        os.replace(tmp, p)
        sizes[str(b)] = int(len(u))
        out.append(int(len(u)))
        print(f"[pool]   bin {b:02d} deduped: {n:,} -> {len(u):,}", flush=True)
    marker.write_text(json.dumps(sizes))
    return out


# ---------------------------------------------------------------- stage D
_SC: dict = {}


def _score_caches(split: str, cd: Path):
    """Per-process SourceCache cache: workers score thousands of bins and
    were re-opening all three caches on every call (measured: ~1-2s/bin of
    pure setup, ~30-60 min over a full stage-D run)."""
    k = (split, str(cd))
    if k not in _SC:
        _SC[k] = tuple(N.SourceCache(cd / f"{split}_source{i}")
                       for i in (1, 2, 3))
    return _SC[k]


def score_bin(task) -> dict:
    b, split, dataset, cache_dir, n = task
    t0 = time.time()
    cd = Path(cache_dir)
    p = np.memmap(cd / f"{split}_pool_bin{b:02d}.p64", dtype=np.int64,
                  mode="r", shape=(n,))
    s1 = (p >> 33).astype(np.int64)
    ot = (p & ((1 << 33) - 1)).astype(np.int64)
    del p
    c1, c2, c3 = _score_caches(split, cd)
    pri, sec = compute_cheap_scores_v2(c1, c2, c3, s1, ot)
    rank = np.zeros(n, np.uint16)
    starts = np.flatnonzero(np.diff(s1)) + 1
    starts = np.concatenate(([0], starts, [n]))
    for g in range(len(starts) - 1):
        lo, hi = int(starts[g]), int(starts[g + 1])
        if hi - lo <= 1:
            rank[lo] = 0
            continue
        # lexicographic: primary desc, secondary desc on ties
        order = np.lexsort((-sec[lo:hi], -pri[lo:hi]))
        r = np.arange(hi - lo, dtype=np.uint32)
        rank[lo + order] = np.minimum(r, 65535).astype(np.uint16)
    for arr, suffix in ((pri, "_score.f32"), (sec, "_score2.f32")):
        tmp = cd / f"{split}_pool_bin{b:02d}{suffix}.tmp"
        arr.astype("<f4").tofile(tmp)
        os.replace(tmp, cd / f"{split}_pool_bin{b:02d}{suffix}")
    rtmp = cd / f"{split}_pool_bin{b:02d}_rank.u16.tmp"
    rank.astype("<u2").tofile(rtmp)
    os.replace(rtmp, cd / f"{split}_pool_bin{b:02d}_rank.u16")
    return {"bin": b, "n": int(n), "s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    N.add_common_args(ap)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=0,
                    help="family workers (stage A); default min(4, cpus)")
    ap.add_argument("--score-workers", type=int, default=0,
                    help="score workers (stage D); default min(6, cpus) — "
                         "measured stage-D ETA ~3.3h @3w, ~1.7h @6w (M4)")
    ap.add_argument("--max-volume", type=int, default=600_000_000,
                    help="full-materialize threshold on post-cap pairs")
    ap.add_argument("--target-volume", type=int, default=500_000_000,
                    help="trim target when over threshold")
    ap.add_argument("--force", action="store_true", help="rebuild pool artifacts")
    ap.add_argument("--rescore", action="store_true",
                    help="redo stage D only (scores/ranks), keep stage A-C")
    args = ap.parse_args()

    workers = args.workers or min(4, _os.cpu_count() or 1)
    score_workers = args.score_workers or min(6, _os.cpu_count() or 1)
    counts_path = args.cache_dir / f"{args.split}_blocking_counts.json"
    if not counts_path.is_file():
        raise SystemExit(f"[pool] missing {counts_path} — run count_families first")
    counts = json.loads(counts_path.read_text())
    total_pc = int(counts["totals"]["post_cap"])

    # ---- decision rule (approved) --------------------------------------
    if total_pc <= args.max_volume:
        mode = "full"
        scale = 1.0
        print(f"[pool] DECISION: FULL — post-cap {total_pc:,} <= "
              f"{args.max_volume:,}", flush=True)
    else:
        mode = "trim"
        scale = min(1.0, args.target_volume / max(total_pc, 1))
        print(f"[pool] DECISION: INFORMED TRIM — post-cap {total_pc:,} > "
              f"{args.max_volume:,}; scale={scale:.4f} -> target "
              f"{args.target_volume:,}", flush=True)
    budgets = {}
    print(f"{'family':7s} {'post_cap':>13s} {'s2_budget':>13s} {'s3_budget':>13s}"
          f"  trim%")
    for fam, e in counts["families"].items():
        b2 = BIG if mode == "full" else max(1, int(e["s2"]["post_cap"] * scale))
        b3 = BIG if mode == "full" else max(1, int(e["s3"]["post_cap"] * scale))
        budgets[fam] = {"s2": b2, "s3": b3}
        trim = 0.0 if mode == "full" else (1 - scale) * 100
        fmt = lambda b: "FULL" if b == BIG else f"{b:,}"
        print(f"{fam:7s} {e['post_cap']:>13,} {fmt(b2):>13} {fmt(b3):>13}"
              f"  {trim:5.1f}%")

    # ensure caches + derive bin table
    caches = N.resolve_split_stems(args.split, args.dataset, args.cache_dir,
                                   limit=args.limit, force=False)
    s1_max = int(caches[1].ids.max()) if caches[1].rows else 0
    n_bins = (s1_max >> BIN_BITS) + 1
    if args.force:
        for p in list(args.cache_dir.glob(f"{args.split}_stageA_fam*")) + \
                 list(args.cache_dir.glob(f"{args.split}_pool_bin*")) + \
                 list(args.cache_dir.glob(f"{args.split}_pool_*.done")):
            p.unlink()
    print(f"[pool] s1_max={s1_max:,} n_bins={n_bins} "
          f"workers A={workers} D={score_workers}", flush=True)

    # ---- stage A --------------------------------------------------------
    # phase-1 gate: counts must match current key emission + cap plan
    if counts.get("key_version") != KEY_VERSION:
        raise SystemExit(
            f"[pool] counts key_version {counts.get('key_version')} != "
            f"{KEY_VERSION} — re-run: python src/count_families.py "
            f"--split {args.split} (it will rebuild keys automatically)")
    caps = counts.get("family_caps") or {
        fam: counts["per_key_cap"] for fam in FAMILIES}
    plan_sig = f"v{KEY_VERSION}:" + json.dumps(caps, sort_keys=True)

    def stage_a_current() -> bool:
        for fi in range(len(FAMILIES)):
            d = args.cache_dir / f"{args.split}_stageA_fam{fi}.done"
            if not d.is_file() or json.loads(d.read_text()).get(
                    "plan_sig") != plan_sig:
                return False
        return True

    if args.rescore:
        if not stage_a_current():
            raise SystemExit(
                "[pool] --rescore: stage A was built for different "
                "keys/caps — run build_pool WITHOUT --rescore (stage A-C "
                "will rebuild, then scores/ranks)")
        print("[pool] --rescore: keeping stage A-C, redoing scores/ranks",
              flush=True)
    elif stage_a_current() and not args.force:
        print("[pool] stage A: up to date", flush=True)
    else:
        print(f"[pool] stage A: materializing {len(FAMILIES)} families "
              f"({workers} workers) ...", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(materialize_family, fi, args.split,
                              str(args.cache_dir),
                              budgets[FAMILIES[fi]],
                              caps[FAMILIES[fi]],
                              args.force, plan_sig)
                    for fi in range(len(FAMILIES))]
            for fut in as_completed(futs):
                r = fut.result()
                tag = "skip" if r.get("skipped") else "done"
                print(f"[pool]   {r['family']:6s} {tag}: "
                      f"{r.get('total_pairs', 0):,} pairs "
                      f"({r.get('elapsed_s', 0)}s)", flush=True)
        print(f"[pool] stage A done ({time.time()-t0:.0f}s)", flush=True)

    # ---- stage B / C -----------------------------------------------------
    if args.rescore:
        marker = args.cache_dir / f"{args.split}_pool_dedup.done"
        if not marker.is_file():
            raise SystemExit("[pool] --rescore requires deduped pool")
        sizes = [json.loads(marker.read_text())[str(b)] for b in range(n_bins)]
        print(f"[pool] --rescore: reusing deduped pool ({sum(sizes):,} pairs)",
              flush=True)
    else:
        scatter_stage_a(args.split, args.cache_dir, n_bins, args.force)
        sizes = dedup_bins(args.split, args.cache_dir, n_bins, args.force)
    total_pool = sum(sizes)
    print(f"[pool] deduped pool total: {total_pool:,} pairs", flush=True)

    # ---- stage D --------------------------------------------------------
    todo = [b for b in range(n_bins)
            if not (args.cache_dir /
                    f"{args.split}_pool_bin{b:02d}_rank.u16").is_file()
            or not (args.cache_dir /
                    f"{args.split}_pool_bin{b:02d}_score.f32").is_file()
            or not (args.cache_dir /
                    f"{args.split}_pool_bin{b:02d}_score2.f32").is_file()]
    if args.force or args.rescore:
        todo = list(range(n_bins))
    if not todo:
        print("[pool] stage D: scores/ranks up to date", flush=True)
    else:
        print(f"[pool] stage D: scoring {len(todo)} bin(s) "
              f"({score_workers} workers) ...", flush=True)
        t0 = time.time()
        tasks = [(b, args.split, str(args.dataset), str(args.cache_dir),
                  sizes[b]) for b in todo]
        done_n = 0
        with ProcessPoolExecutor(max_workers=score_workers) as ex:
            futs = [ex.submit(score_bin, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                done_n += 1
                el = time.time() - t0
                eta = el / done_n * (len(todo) - done_n)
                print(f"[pool]   bin {r['bin']:02d} scored: {r['n']:,} pairs "
                      f"({r['s']}s, {done_n}/{len(todo)}, ETA {eta:.0f}s)",
                      flush=True)
        print(f"[pool] stage D done ({time.time()-t0:.0f}s)", flush=True)

    # ---- meta -----------------------------------------------------------
    meta = {
        "split": args.split,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode,
        "ranker": "lex-jaccard-translit-v2",
        "decision": {"post_cap_total": total_pc,
                     "max_volume": args.max_volume,
                     "target_volume": args.target_volume,
                     "scale": scale,
                     "budgets": budgets},
        "per_key_cap": counts["per_key_cap"],
        "family_caps": caps,
        "plan_sig": plan_sig,
        "key_version": KEY_VERSION,
        "bin_bits": BIN_BITS,
        "s1_max": s1_max,
        "n_bins": n_bins,
        "bins": [{"n": int(s)} for s in sizes],
        "pool_total": int(total_pool),
        "counts_file": counts_path.name,
    }
    out = args.cache_dir / f"{args.split}_pool_meta.json"
    out.write_text(json.dumps(meta, indent=1))
    print(f"[pool] wrote {out}", flush=True)
    print(f"[pool] ANNOTATED POOL: {total_pool:,} pairs over {n_bins} bin(s) "
          f"— next: analyze_misses.py", flush=True)


if __name__ == "__main__":
    main()
