#!/usr/bin/env python3
"""trace_misses.py — Phase 1, Step 1: failure-point tracer for stage-A misses.

For EVERY GT pair absent from the annotated pool, walk the key pipeline the
same way blocking does and record exactly where it was lost:

  CAP-VICTIM   some family emitted the same key on both rows, but that
               bucket's cartesian product exceeds per-key-cap (5000), so
               try1's semantics dropped the whole bucket. Records the
               MINIMUM bucket size over all shared keys (drives the Step-2
               cap curve: a pair survives cap X iff min_pc <= X).
  BUG          some family emitted a shared key whose bucket is WITHIN the
               cap (and budget is full) — the pair should be in the pool and
               isn't. This is build-path breakage; examples are printed with
               full key payloads.
  NOKEY        no family shares any key. Per-family emission columns say
               why (emitted on side1 only / side2 only / both but disjoint /
               neither).

Outputs:
  output/trace_report.txt          tables + examples
  cache/{split}_trace.npz          capv_packed + capv_minpc (for cap_curve)
  cache/{split}_trace_summary.json reasons, histograms, matrices

Run after build_pool (needs the pool) and BEFORE enable_translit (keys must
match the pool's generation). ~5-8 min with --workers 4.
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
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import normalize as N
from blocking import FAMILIES

NOKEY_COLS = ["s1_only", "s2_only", "both_disjoint", "neither"]
CAP_HIST_EDGES = [(0, 7500), (7500, 20000), (20000, 50000), (50000, 200000),
                  (200000, 1 << 62)]


# ---------------------------------------------------------------- keys
def row_family_keys(cache: N.SourceCache, row: int, s3: bool
                    ) -> Dict[int, List[Tuple[int, str]]]:
    """Exactly the key payloads build_all_keys would emit for this row."""
    a = cache.arrays
    c = int(a["country"][row])
    ns = a["name_sorted"][row].decode("utf-8", "replace")
    nc = a["name_core"][row].decode("utf-8", "replace")
    nonlat = int(a["name_script"][row]) != 0
    out: Dict[int, List[Tuple[int, str]]] = {f: [] for f in range(len(FAMILIES))}
    if not ns:
        return out
    sig_payload = ns
    core_toks = nc.split()
    sorted_toks = ns.split()
    if nonlat:
        tr = a["name_translit"][row].decode("utf-8", "replace")
        if tr and tr != ns:
            sig_payload = tr
            t_toks = tr.split()
        else:
            t_toks = []
    else:
        t_toks = []

    def add(fam: int, payload: str) -> None:
        out[fam].append((N.h64(f"{c}|{fam}|{payload}"), payload))

    add(0, sig_payload)
    if sorted_toks:
        add(1, " ".join(sorted_toks[:3] + sorted_toks[-1:]))
        if nonlat and t_toks:
            add(1, " ".join(t_toks[:3] + t_toks[-1:]))
    for t in core_toks:
        if len(t) >= 3 and t not in N.STOP_NAME_TOKENS:
            add(2, t)
    for t in t_toks:
        if len(t) >= 3 and t not in N.STOP_NAME_TOKENS:
            add(2, t)
    ph_src = t_toks if (nonlat and t_toks) else core_toks
    if len(ph_src) >= 2:
        add(3, N.phon_key(ph_src[0]) + " " + N.phon_key(ph_src[-1]))
    elif ph_src:
        add(3, N.phon_key(ph_src[0]))
    if len(ph_src) >= 3:
        ini = "".join(t[0] for t in ph_src if t and t[0].isalpha())
        if len(ini) >= 3:
            add(4, ini)
    hn = a["houseno"][row].decode("utf-8", "replace")
    if hn:
        add(5, hn)
        if core_toks and core_toks[0]:
            add(7, hn + "|" + core_toks[0])
        st = a["state"][row].decode("utf-8", "replace")
        if st:
            add(8, hn + "|" + st)
    at = a["addr_tokens"][row].decode("utf-8", "replace")
    if at:
        toks = at.split()
        for t in toks:
            if len(t) >= 4 and t.isalpha() and t not in N.ADDR_STOP_TOKENS:
                add(6, t)
        for j in range(len(toks) - 1):
            add(9, toks[j] + " " + toks[j + 1])
    return out


# ---------------------------------------------------------------- worker
def trace_chunk(task) -> dict:
    (split, cache_dir, r1, roth, is3, s1_ids, ot_ids, per_key_cap,
     u_paths, active) = task
    cd = Path(cache_dir)
    caches = {i: N.SourceCache(cd / f"{split}_source{i}") for i in (1, 2, 3)}
    u_arrs = {}
    for fi in active:
        for s in (1, 2, 3):
            u = np.load(u_paths[fi][s - 1][0], mmap_mode="r")
            c = np.load(u_paths[fi][s - 1][1], mmap_mode="r")
            u_arrs[(fi, s)] = (u, c)

    def count_of(fi: int, side: int, key: int) -> int:
        u, c = u_arrs[(fi, side)]
        i = int(np.searchsorted(u, np.int64(key)))
        if i < len(u) and int(u[i]) == key:
            return int(c[i])
        return 0

    reasons = Counter()
    cap_hist = Counter()
    nokey_matrix = np.zeros((len(FAMILIES), len(NOKEY_COLS)), np.int64)
    cap_family = Counter()
    capv_packed, capv_pc, capv_fam = [], [], []
    examples = {"BUG": [], "CAP": [], "NOKEY": []}
    for j in range(len(r1)):
        row1 = int(r1[j])
        rowo = int(roth[j])
        s3 = bool(is3[j])
        if row1 < 0 or rowo < 0:
            reasons["NOKEY"] += 1
            continue
        other_side = 3 if s3 else 2
        k1 = row_family_keys(caches[1], row1, False)
        ko = row_family_keys(caches[other_side], rowo, s3)
        best_pc = None
        best_fam = None
        shared_any = False
        for fi in active:
            keys1 = dict(k1[fi])
            keyso = dict(ko[fi])
            shared = set(keys1) & set(keyso)
            if not shared:
                continue
            shared_any = True
            for key in shared:
                pc = (count_of(fi, 1, key)
                      * count_of(fi, other_side, key))
                if best_pc is None or pc < best_pc:
                    best_pc, best_fam = pc, fi
        if best_pc is not None and best_pc <= per_key_cap:
            reasons["BUG"] += 1
            if len(examples["BUG"]) < 40:
                shared = (set(dict(k1[best_fam])) & set(dict(ko[best_fam])))
                pl1 = dict(k1[best_fam]).get(next(iter(shared)), "?") \
                    if shared else "?"
                plo = dict(ko[best_fam]).get(next(iter(shared)), "?") \
                    if shared else "?"
                examples["BUG"].append(
                    {"s1": int(s1_ids[j]), "oth": int(ot_ids[j]),
                     "family": FAMILIES[best_fam], "bucket_pairs": int(best_pc),
                     "payload_s1": pl1, "payload_oth": plo})
        elif shared_any:
            reasons["CAP"] += 1
            cap_family[FAMILIES[best_fam]] += 1
            for lo_e, hi_e in CAP_HIST_EDGES:
                if lo_e < best_pc <= hi_e:
                    cap_hist[f"{lo_e}-{hi_e}"] += 1
                    break
            capv_packed.append((int(s1_ids[j]) << 33) | int(ot_ids[j]))
            capv_pc.append(int(best_pc))
            capv_fam.append(int(best_fam))
            if len(examples["CAP"]) < 40:
                examples["CAP"].append(
                    {"s1": int(s1_ids[j]), "oth": int(ot_ids[j]),
                     "family": FAMILIES[best_fam],
                     "bucket_pairs": int(best_pc)})
        else:
            reasons["NOKEY"] += 1
            for fi in active:
                e1 = len(k1[fi]) > 0
                eo = len(ko[fi]) > 0
                if e1 and not eo:
                    nokey_matrix[fi, 0] += 1
                elif eo and not e1:
                    nokey_matrix[fi, 1] += 1
                elif e1 and eo:
                    nokey_matrix[fi, 2] += 1
                else:
                    nokey_matrix[fi, 3] += 1
            if len(examples["NOKEY"]) < 40:
                examples["NOKEY"].append(
                    {"s1": int(s1_ids[j]), "oth": int(ot_ids[j]),
                     "s1_toks": dict(k1[2]), "oth_toks": dict(ko[2]),
                     "s1_atok": dict(k1[6]), "oth_atok": dict(ko[6])})
    return {"reasons": dict(reasons), "cap_hist": dict(cap_hist),
            "cap_family": dict(cap_family),
            "nokey_matrix": nokey_matrix,
            "capv_packed": np.asarray(capv_packed, np.int64),
            "capv_pc": np.asarray(capv_pc, np.int64),
            "capv_fam": np.asarray(capv_fam, np.uint8),
            "examples": examples}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    N.add_common_args(ap)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="rebuild per-key count arrays")
    args = ap.parse_args()
    workers = args.workers or min(4, _os.cpu_count() or 1)
    t0 = time.time()
    cd = args.cache_dir
    meta = json.loads((cd / f"{args.split}_pool_meta.json").read_text())
    counts = json.loads((cd / f"{args.split}_blocking_counts.json").read_text())
    cap = int(counts["per_key_cap"])
    bin_bits = int(meta["bin_bits"])
    n_bins = int(meta["n_bins"])

    # ---- GT + miss set ----
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
    found = np.zeros(len(p_gt), bool)
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
        found[lo:hi] = (pos < nb) & (pp[cp] == gp)
        del pp
    miss = np.flatnonzero(~found)
    n_miss = len(miss)
    print(f"[trace] stage-A misses: {n_miss:,} "
          f"of {len(p_gt):,} ({time.time()-t0:.0f}s)", flush=True)

    # ---- per-family key->count arrays (for bucket sizes) ----
    kdir = cd / "pool_keys"
    tdir = cd / "trace_keys"
    tdir.mkdir(exist_ok=True)
    u_paths = {}
    active = []
    for fi in range(len(FAMILIES)):
        # a family with no key files is not part of this pool's key
        # generation (e.g. tracing an old 8-family pool with
        # KEY_VERSION-2 scripts) — skip it entirely, BEFORE touching disk.
        kfiles = [kdir / f"{args.split}_f{fi}_s{s}_keys.npy"
                  for s in (1, 2, 3)]
        if not all(k.is_file() for k in kfiles):
            continue
        row = []
        for s in (1, 2, 3):
            up = tdir / f"{args.split}_f{fi}_s{s}_u.npy"
            cp_ = tdir / f"{args.split}_f{fi}_s{s}_c.npy"
            if args.force or not (up.is_file() and cp_.is_file()):
                k = np.load(kdir / f"{args.split}_f{fi}_s{s}_keys.npy",
                            mmap_mode="r")
                u, c = np.unique(k, return_counts=True)
                np.save(up, u)
                np.save(cp_, c)
            row.append((up, cp_))
        active.append(fi)
        u_paths[fi] = row
    if len(active) < len(FAMILIES):
        print(f"[trace] NOTE: only families "
              f"{[FAMILIES[fi] for fi in active]} have key files "
              f"(pool built pre-KEY_VERSION-2?) — the rest are inactive",
              flush=True)
    print(f"[trace] key-count arrays ready ({time.time()-t0:.0f}s)", flush=True)

    # ---- row maps for misses ----
    caches = {i: N.SourceCache(cd / f"{args.split}_source{i}")
              for i in (1, 2, 3)}
    r1 = caches[1].row_of(gt_s1[miss])
    m2 = gt_ot[miss] < N.S3_OFFSET
    roth = np.zeros(n_miss, np.int64)
    if m2.any():
        roth[m2] = caches[2].row_of(gt_ot[miss][m2])
    if (~m2).any():
        roth[~m2] = caches[3].row_of(gt_ot[miss][~m2] - N.S3_OFFSET)
    bad = (r1 < 0) | (roth < 0)
    if bad.any():
        print(f"[trace] WARNING: {int(bad.sum())} misses with rows not in "
              f"cache — counted as NOKEY")
    s1_ids = gt_s1[miss]
    ot_ids = gt_ot[miss]

    # ---- parallel trace ----
    n_chunks = workers * 4
    bounds = np.linspace(0, n_miss, n_chunks + 1).astype(int)
    tasks = []
    for w in range(n_chunks):
        sl = slice(int(bounds[w]), int(bounds[w + 1]))
        if sl.start == sl.stop:
            continue
        tasks.append((args.split, str(cd), r1[sl], roth[sl], ~m2[sl],
                      s1_ids[sl], ot_ids[sl], cap, u_paths, active))
    agg_reasons, agg_hist, agg_fam = Counter(), Counter(), Counter()
    agg_mat = np.zeros((len(FAMILIES), len(NOKEY_COLS)), np.int64)
    all_packed, all_pc, all_fam = [], [], []
    ex = {"BUG": [], "CAP": [], "NOKEY": []}
    with ProcessPoolExecutor(max_workers=workers) as pex:
        futs = [pex.submit(trace_chunk, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            agg_reasons.update(r["reasons"])
            agg_hist.update(r["cap_hist"])
            agg_fam.update(r["cap_family"])
            agg_mat += r["nokey_matrix"]
            all_packed.append(r["capv_packed"])
            all_pc.append(r["capv_pc"])
            all_fam.append(r["capv_fam"])
            for k in ex:
                room = 40 - len(ex[k])
                if room > 0:
                    ex[k].extend(r["examples"][k][:room])
            done += 1
            print(f"[trace] chunk {done}/{len(tasks)} ({time.time()-t0:.0f}s)",
                  flush=True)

    # ---- report ----
    lines = []

    def L(s=""):
        lines.append(s)
        print(s, flush=True)

    n = max(sum(agg_reasons.values()), 1)
    L("=" * 72)
    L(f"STAGE-A MISS TRACER — split={args.split} — {n_miss:,} misses")
    L("=" * 72)
    L(f"  {'reason':10s} {'count':>10s} {'%':>8s}   meaning")
    L(f"  {'CAP':10s} {agg_reasons.get('CAP',0):>10,} "
      f"{agg_reasons.get('CAP',0)/n*100:>7.2f}%   bucket > per-key-cap "
      f"({cap:,}) — whole bucket dropped")
    L(f"  {'NOKEY':10s} {agg_reasons.get('NOKEY',0):>10,} "
      f"{agg_reasons.get('NOKEY',0)/n*100:>7.2f}%   no family shares any key")
    L(f"  {'BUG':10s} {agg_reasons.get('BUG',0):>10,} "
      f"{agg_reasons.get('BUG',0)/n*100:>7.2f}%   shared key WITHIN cap "
      f"but pair absent — build-path breakage")
    L()
    L("CAP-VICTIM min bucket size histogram (drives cap curve):")
    for lo_e, hi_e in CAP_HIST_EDGES:
        key = f"{lo_e}-{hi_e}"
        L(f"  {key:>16s}: {agg_hist.get(key,0):>10,}")
    L("  best (smallest) family per CAP victim:")
    for fam, c_ in agg_fam.most_common():
        L(f"    {fam:7s}: {c_:>10,}")
    L()
    L("NOKEY per-family emission (rows=family, cols="
      + "/".join(NOKEY_COLS) + "):")
    L(f"  {'family':8s} " + " ".join(f"{c:>14s}" for c in NOKEY_COLS))
    for fi, fam in enumerate(FAMILIES):
        L(f"  {fam:8s} " + " ".join(f"{agg_mat[fi, ci]:>14,}"
                                    for ci in range(len(NOKEY_COLS))))
    L()
    for tag in ("BUG", "CAP", "NOKEY"):
        if ex[tag]:
            L(f"--- examples [{tag}] (up to {len(ex[tag])}) ---")
            for e in ex[tag][:12]:
                if tag == "BUG":
                    L(f"  S1-{e['s1']} x {e['oth']}  fam={e['family']} "
                      f"bucket={e['bucket_pairs']:,}  "
                      f"s1_key='{e['payload_s1']}' oth_key='{e['payload_oth']}'")
                elif tag == "CAP":
                    L(f"  S1-{e['s1']} x {e['oth']}  best_family="
                      f"{e['family']} min_bucket={e['bucket_pairs']:,}")
                else:
                    L(f"  S1-{e['s1']} x {e['oth']}  "
                      f"s1_tok={list(e['s1_toks'].values())[:6]} "
                      f"oth_tok={list(e['oth_toks'].values())[:6]} "
                      f"s1_atok={list(e['s1_atok'].values())[:5]} "
                      f"oth_atok={list(e['oth_atok'].values())[:5]}")
            L()

    packed = (np.concatenate(all_packed) if all_packed
              else np.empty(0, np.int64))
    pc = np.concatenate(all_pc) if all_pc else np.empty(0, np.int64)
    fam = (np.concatenate(all_fam) if all_fam
           else np.empty(0, np.uint8))
    np.savez(cd / f"{args.split}_trace.npz", capv_packed=packed,
             capv_minpc=pc, capv_fam=fam)
    summary = {"split": args.split, "per_key_cap": cap,
               "n_gt": int(len(p_gt)), "n_miss": int(n_miss),
               "reasons": dict(agg_reasons),
               "cap_hist": dict(agg_hist),
               "cap_family": dict(agg_fam),
               "nokey_matrix": agg_mat.tolist(),
               "nokey_cols": NOKEY_COLS}
    (cd / f"{args.split}_trace_summary.json").write_text(
        json.dumps(summary, indent=1))
    out_dir = N.TRY2_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "trace_report.txt").write_text("\n".join(lines) + "\n")
    L(f"[trace] wrote output/trace_report.txt + {args.split}_trace.npz")
    L(f"[trace] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
