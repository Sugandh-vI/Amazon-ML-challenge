#!/usr/bin/env python3
"""blocking.py — candidate generation for try1.

Two stages, exactly as specified in candidate_pairs.tsv:

Stage A — multi-key union.  Eight independent key families (all keys are
country-prefixed and 64-bit hashed):

    sig    token-sorted suffix-stripped name          (exact after normalize)
    pfx    first-3 + last tokens of sorted name       (order/partial tolerant)
    tok    every rare-ish core name token             (containment / DBA / fka)
    phon   phonetic key of first+last core token      (typos / transliteration)
    ini    initials of >=3 core tokens                (IAG <- Indian Ace Global)
    hn     house number                               (address bridge)
    atok   address token (len>=4)                     (locality / street bridge)
    combo  house number + first name token            (selective address+name)

Non-Latin rows additionally emit sig/pfx/phon/ini keys built from the
transliterated name (and tok keys from transliterated tokens).  Every key is
prefixed with country.  Buckets whose cartesian product exceeds --per-key-cap
are dropped; total stage-A pairs are bounded by --budget (largest buckets
dropped first).

Stage B — per-S1 capping.  Each S1 keeps at most --k candidates ranked by a
cheap name score  max(token_jaccard, token_containment)  on token-sorted core
names.  The output of stage B *is* candidate_pairs.tsv — the exact set the
matching model scores.

Outputs (in --cache-dir):
    candidates_{split}.npz   CSR arrays: s1 ids, indptr, other ids
    blocking_report_{split}.txt
and, for --split test, ../output/candidate_pairs.tsv.
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
from typing import Dict, List, Optional, Tuple

import numpy as np

import normalize as N
from normalize import S3_OFFSET
import features as F

FAMILIES = ["sig", "pfx", "tok", "phon", "ini", "hn", "atok", "combo",
            "hnstate", "atok2"]
# bump whenever key emission changes -> count_families auto-rebuilds keys,
# build_pool invalidates stage A via plan_sig (embeds KEY_VERSION).
KEY_VERSION = 2

# --------------------------------------------------------------------------
# Stage A: key building
# --------------------------------------------------------------------------
def build_all_keys(cache: N.SourceCache, s3: bool,
                   only: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One streaming pass over a cache -> (keys i64, entity ids i64, family u1).

    ``only`` restricts emitted keys to a single family (saves hashing work when
    blocking runs family-by-family to keep peak RAM low).
    """
    a = cache.arrays
    n = cache.rows
    keys: List[int] = []
    ents: List[int] = []
    fams: List[int] = []
    raw_ids = cache.ids
    for i in range(n):
        c = int(a["country"][i])
        ns = a["name_sorted"][i].decode("utf-8", "replace")
        nc = a["name_core"][i].decode("utf-8", "replace")
        nonlat = int(a["name_script"][i]) != 0
        ent = int(raw_ids[i]) + (S3_OFFSET if s3 else 0)
        if not ns:
            continue
        sig_payload = ns
        core_toks = nc.split()
        sorted_toks = ns.split()
        if nonlat:
            tr = a["name_translit"][i].decode("utf-8", "replace")
            if tr and tr != ns:
                sig_payload = tr
                t_toks = tr.split()
            else:
                t_toks = []
        else:
            t_toks = []

        def add(fam: int, payload: str) -> None:
            if only is not None and fam != only:
                return
            keys.append(N.h64(f"{c}|{fam}|{payload}"))
            ents.append(ent)
            fams.append(fam)

        # sig / pfx
        add(0, sig_payload)
        if sorted_toks:
            pfx = " ".join(sorted_toks[:3] + sorted_toks[-1:])
            add(1, pfx)
            if nonlat and t_toks:
                add(1, " ".join(t_toks[:3] + t_toks[-1:]))
        # tok (raw core tokens + translit tokens for non-latin rows)
        for t in core_toks:
            if len(t) >= 3 and t not in N.STOP_NAME_TOKENS:
                add(2, t)
        for t in t_toks:
            if len(t) >= 3 and t not in N.STOP_NAME_TOKENS:
                add(2, t)
        # phon / ini  (translit-derived for non-latin rows)
        ph_src = t_toks if (nonlat and t_toks) else core_toks
        if len(ph_src) >= 2:
            add(3, N.phon_key(ph_src[0]) + " " + N.phon_key(ph_src[-1]))
        elif ph_src:
            add(3, N.phon_key(ph_src[0]))
        if len(ph_src) >= 3:
            ini = "".join(t[0] for t in ph_src if t and t[0].isalpha())
            if len(ini) >= 3:
                add(4, ini)
        # address keys
        hn = a["houseno"][i].decode("utf-8", "replace")
        if hn:
            add(5, hn)
            if core_toks and core_toks[0]:
                add(7, hn + "|" + core_toks[0])
            # KEY_VERSION 2: houseno|state — HN_AND_STATE was the largest
            # stage-A-miss family (451k pairs); hn alone has bucket products
            # 11k-623k (always capped); hn+state splits it into small buckets.
            st = a["state"][i].decode("utf-8", "replace")
            if st:
                add(8, hn + "|" + st)
        at = a["addr_tokens"][i].decode("utf-8", "replace")
        if at:
            toks = at.split()
            for t in toks:
                if len(t) >= 4 and t.isalpha() and t not in N.ADDR_STOP_TOKENS:
                    add(6, t)
            # KEY_VERSION 2: adjacent addr-token bigrams — single atok tokens
            # ('street','road',...) carried products up to billions and were
            # the best family for 712k CAP victims; bigrams keep the pair,
            # shrink the bucket.
            for j in range(len(toks) - 1):
                add(9, toks[j] + " " + toks[j + 1])
    return (np.asarray(keys, dtype=np.int64),
            np.asarray(ents, dtype=np.int64),
            np.asarray(fams, dtype=np.uint8))

def expand_join(k1, e1, k2, e2, per_key_cap: int, budget: int):
    """Sort-merge join with per-bucket and total pair budgets."""
    o1 = np.argsort(k1, kind="stable")
    s1k, s1e = k1[o1], e1[o1]
    o2 = np.argsort(k2, kind="stable")
    s2k, s2e = k2[o2], e2[o2]
    u1, st1, c1 = np.unique(s1k, return_index=True, return_counts=True)
    u2, st2, c2 = np.unique(s2k, return_index=True, return_counts=True)
    common, i1, i2 = np.intersect1d(u1, u2, return_indices=True)
    if len(common) == 0:
        return (np.empty(0, e1.dtype), np.empty(0, e2.dtype), 0, 0, 0)
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
    out1 = np.empty(total, dtype=e1.dtype)
    out2 = np.empty(total, dtype=e2.dtype)
    pos = 0
    for b in range(len(common)):
        a_st, a_n = int(st1[i1[b]]), int(c1[i1[b]])
        b_st, b_n = int(st2[i2[b]]), int(c2[i2[b]])
        ids_a = s1e[a_st:a_st + a_n]
        ids_b = s2e[b_st:b_st + b_n]
        out1[pos:pos + a_n * b_n] = np.repeat(ids_a, b_n)
        out2[pos:pos + a_n * b_n] = np.tile(ids_b, a_n)
        pos += a_n * b_n
    return out1, out2, dropped_cap, dropped_budget, len(common)

# --------------------------------------------------------------------------
# Stage B + outputs
# --------------------------------------------------------------------------
def topk_mask(cheap: np.ndarray, s1_sorted: np.ndarray, k: int) -> np.ndarray:
    """True for pairs kept as one of their S1's top-k cheap-score candidates."""
    mask = np.zeros(len(cheap), dtype=bool)
    if len(s1_sorted) == 0:
        return mask
    starts = np.flatnonzero(np.diff(s1_sorted)) + 1
    starts = np.concatenate(([0], starts, [len(s1_sorted)]))
    for g in range(len(starts) - 1):
        lo, hi = int(starts[g]), int(starts[g + 1])
        cnt = hi - lo
        if cnt <= k:
            mask[lo:hi] = True
        else:
            seg = cheap[lo:hi]
            top = np.argpartition(seg, -k)[-k:]
            mask[lo + top] = True
    return mask

def csr_from_mask(s1: np.ndarray, other: np.ndarray, mask: np.ndarray):
    s1 = s1[mask]
    other = other[mask]
    if len(s1) == 0:
        return (np.empty(0, np.int64), np.array([0], np.int64),
                np.empty(0, np.int64))
    order = np.lexsort((other, s1))
    s1, other = s1[order], other[order]
    uniq, starts, counts = np.unique(s1, return_index=True, return_counts=True)
    indptr = np.zeros(len(uniq) + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    return uniq, indptr, other

def write_candidate_tsv(path: Path, s1_all_ids: np.ndarray,
                        s1: np.ndarray, indptr: np.ndarray, other: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_zero = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for ent in s1_all_ids:
            pos = int(np.searchsorted(s1, ent))
            if pos < len(s1) and s1[pos] == ent:
                lo, hi = int(indptr[pos]), int(indptr[pos + 1])
                line = ",".join(N.decode_other(int(c)) for c in other[lo:hi])
            else:
                line = ""
                n_zero += 1
            f.write(f"S1-{int(ent)}\t{line}\n")
    print(f"[blocking] wrote {path} ({n_zero:,} S1 entities with zero candidates)",
          flush=True)

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def pack_pairs(s1: np.ndarray, other: np.ndarray) -> np.ndarray:
    if len(s1) and int(s1.max()) >= (1 << 31):
        raise ValueError("S1 id >= 2**31 — packing scheme needs revisiting")
    if len(other) and int(other.max()) >= (1 << 32):
        raise ValueError("other id >= 2**32 — packing scheme needs revisiting")
    return (s1.astype(np.int64) << 33) | other.astype(np.int64)

def blocking_report(split, dataset, cache_dir, caches, s1, indptr, other,
                    fam_stats, stage_a_pairs, args) -> str:
    n_s1 = caches[1].rows
    n_other = caches[2].rows + caches[3].rows
    total = len(other)
    counts = np.diff(indptr) if len(s1) else np.array([], np.int64)
    lines = []
    lines.append(f"=== blocking report: split={split} ===")
    lines.append(f"records: S1={n_s1:,}  S2={caches[2].rows:,}  S3={caches[3].rows:,}")
    lines.append(f"stage-A families:")
    for name, st in fam_stats.items():
        lines.append(
            f"  {name:6s} pairs={st['pairs']:>12,}  buckets_kept={st['buckets']:>10,} "
            f"dropped_cap={st['drop_cap']:>9,} dropped_budget={st['drop_budget']:>9,}")
    lines.append(f"stage-A unique pairs: {stage_a_pairs:,}")
    lines.append(f"stage-B candidates:   {total:,} "
                 f"(top-{args.k} cap per S1)")
    if n_s1:
        lines.append(f"candidates per S1: mean={counts.mean() if len(counts) else 0:.1f} "
                     f"median={np.median(counts) if len(counts) else 0:.0f} "
                     f"p95={np.percentile(counts, 95) if len(counts) else 0:.0f} "
                     f"max={counts.max() if len(counts) else 0}")
        lines.append(f"S1 with zero candidates: {n_s1 - len(s1):,} "
                     f"({100.0 * (n_s1 - len(s1)) / n_s1:.2f}%)")
        rr = total / float(n_s1 * n_other)
        lines.append(f"reduction ratio: {rr:.3e}  "
                     f"(eliminated {100.0 * (1 - rr):.4f}% of the cross-product)")

    gt_path = dataset / split / f"{split}_ground_truth.tsv"
    if gt_path.is_file():
        gt_s1, gt_other = [], []
        for s1e, ids in N.stream_ground_truth(gt_path):
            for o in ids:
                gt_s1.append(s1e)
                gt_other.append(o)
        gt_s1 = np.asarray(gt_s1, np.int64)
        gt_other = np.asarray(gt_other, np.int64)
        gt_pack = pack_pairs(gt_s1, gt_other)
        if total:
            s1_flat = np.repeat(s1, np.diff(indptr))
            cand_pack = pack_pairs(s1_flat, other)
        else:
            cand_pack = np.empty(0, np.int64)
        hit = np.isin(gt_pack, cand_pack)
        lines.append(f"--- recall vs ground truth ({len(gt_pack):,} true pairs) ---")
        lines.append(f"pair recall:        {hit.mean():.4%}")
        # entity-level (per_ent_full indexed POSITIONALLY over sorted unique ids)
        gt_ent = np.unique(gt_s1)
        per_ent_full = np.zeros(len(gt_ent), bool)
        order = np.argsort(gt_s1, kind="stable")
        g_sorted = gt_s1[order]
        h_sorted = hit[order]
        starts = np.flatnonzero(np.diff(g_sorted)) + 1
        starts = np.concatenate(([0], starts, [len(g_sorted)]))
        for a, b in zip(starts[:-1], starts[1:]):
            pos = int(np.searchsorted(gt_ent, g_sorted[a]))
            per_ent_full[pos] = h_sorted[a:b].all()
        lines.append(f"entity recall (ALL matches found): {per_ent_full.mean():.4%}")
        # by country
        rows1 = caches[1].row_of(gt_ent)
        cty = caches[1].get("country", rows1)
        for code, label in N.COUNTRY_LABELS.items():
            sel = cty == code
            if sel.any():
                lines.append(f"  entity recall [{label}]: "
                             f"{per_ent_full[sel].mean():.4%} "
                             f"(n={int(sel.sum()):,})")
        # cross-script segment: pairs whose counterpart name is non-Latin
        other_nonlat = np.zeros(len(gt_other), bool)
        for cache, sel, off in ((caches[2], gt_other < S3_OFFSET, 0),
                                (caches[3], gt_other >= S3_OFFSET, S3_OFFSET)):
            if not sel.any():
                continue
            sub = np.flatnonzero(sel)
            rows = cache.row_of(gt_other[sub] - off)
            okr = rows >= 0
            nl = np.zeros(len(sub), bool)
            nl[okr] = cache.get("name_nonascii", rows[okr]) > 0
            other_nonlat[sub] = nl
        seg = other_nonlat
        if seg.any():
            lines.append(f"  pair recall [non-Latin counterpart name]: {hit[seg].mean():.4%} "
                         f"({int(seg.sum()):,} pairs)")
        if (~seg).any():
            lines.append(f"  pair recall [Latin counterpart name]:      {hit[~seg].mean():.4%} "
                         f"({int((~seg).sum()):,} pairs)")
    else:
        lines.append("(no ground truth for this split — recall not computed)")
    report = "\n".join(lines)
    print(report, flush=True)
    out = cache_dir / f"blocking_report_{split}.txt"
    out.write_text(report)
    print(f"[blocking] report saved to {out}", flush=True)
    return report

# --------------------------------------------------------------------------
def run_blocking(split: str, dataset: Path, cache_dir: Path, args) -> Dict:
    t0 = time.time()
    caches = N.resolve_split_stems(split, dataset, cache_dir,
                                   limit=args.limit, force=args.force_cache)
    print("[blocking] stage A: one key family at a time (bounded peak RAM)",
          flush=True)
    fam_stats: Dict[str, dict] = {}
    pair_chunks: List[Tuple[np.ndarray, np.ndarray]] = []
    per_family_budget = max(1, args.budget // (len(FAMILIES) * 2))
    for fi, fname in enumerate(FAMILIES):
        t_f = time.time()
        k1, e1, f1 = build_all_keys(caches[1], s3=False, only=fi)
        m1 = f1 == fi
        stats = {"pairs": 0, "buckets": 0, "drop_cap": 0, "drop_budget": 0}
        p1_all, p2_all = [], []
        for cache_i in (2, 3):
            ke, ee, fe = build_all_keys(caches[cache_i],
                                        s3=(cache_i == 3), only=fi)
            me = fe == fi
            if m1.any() and me.any():
                p1, p2, dc, db, nb = expand_join(
                    k1[m1], e1[m1], ke[me], ee[me],
                    args.per_key_cap, per_family_budget)
                stats["drop_cap"] += dc
                stats["drop_budget"] += db
                stats["buckets"] += nb
                stats["pairs"] += len(p1)
                if len(p1):
                    p1_all.append(p1)
                    p2_all.append(p2)
            del ke, ee, fe
        del k1, e1, f1
        fam_stats[fname] = stats
        print(f"[blocking]   family {fname:6s}: pairs={stats['pairs']:,} "
              f"buckets={stats['buckets']:,} "
              f"(dropped cap={stats['drop_cap']:,} "
              f"budget={stats['drop_budget']:,}, {time.time()-t_f:.0f}s)",
              flush=True)
        if p1_all:
            pair_chunks.append((np.concatenate(p1_all), np.concatenate(p2_all)))
        del p1_all, p2_all

    if pair_chunks:
        all_s1 = np.concatenate([c[0] for c in pair_chunks])
        all_ot = np.concatenate([c[1] for c in pair_chunks])
        del pair_chunks
        packed = np.unique(pack_pairs(all_s1, all_ot))
        del all_s1, all_ot
        s1_u = packed >> 33
        ot_u = packed & ((1 << 33) - 1)
        del packed
        order = np.argsort(s1_u, kind="stable")
        s1_u, ot_u = s1_u[order], ot_u[order]
    else:
        s1_u = np.empty(0, np.int64)
        ot_u = np.empty(0, np.int64)
    stage_a = len(s1_u)
    print(f"[blocking] stage-A unique pairs: {stage_a:,} ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- stage B: cheap score + top-k -------------------------------------
    print("[blocking] stage B: cheap scoring + top-k capping ...", flush=True)
    cheap = F.compute_cheap_scores(caches[1], caches[2], caches[3], s1_u, ot_u)
    keep = topk_mask(cheap, s1_u, args.k)
    s1_c, indptr, other_c = csr_from_mask(s1_u, ot_u, keep)
    del cheap, keep, s1_u, ot_u
    print(f"[blocking] stage-B candidates: {len(other_c):,} "
          f"({time.time()-t0:.0f}s total)", flush=True)

    np.savez_compressed(cache_dir / f"candidates_{split}.npz",
                        s1=s1_c, indptr=indptr, other=other_c, k=args.k)
    (cache_dir / f"candidates_{split}.meta.json").write_text(json.dumps({
        "split": split, "k": args.k, "per_key_cap": args.per_key_cap,
        "budget": args.budget, "stage_a_pairs": stage_a,
        "candidates": int(len(other_c)),
    }))

    blocking_report(split, dataset, cache_dir, caches, s1_c, indptr, other_c,
                    fam_stats, stage_a, args)

    if split == "test":
        out = N.TRY1_DIR / "output" / "candidate_pairs.tsv"
        write_candidate_tsv(out, caches[1].ids, s1_c, indptr, other_c)
    return {"s1": s1_c, "indptr": indptr, "other": other_c}

def load_candidates(cache_dir: Path, split: str):
    z = np.load(cache_dir / f"candidates_{split}.npz")
    return z["s1"], z["indptr"], z["other"]

def main():
    ap = argparse.ArgumentParser(description="try1 blocking / candidate generation")
    N.add_common_args(ap)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("-k", "--k", type=int, default=80,
                    help="max candidates kept per S1 (default %(default)s)")
    ap.add_argument("--per-key-cap", type=int, default=5000,
                    help="drop key buckets whose cartesian product exceeds this")
    ap.add_argument("--budget", type=int, default=200_000_000,
                    help="stage-A total pair budget (default %(default)s)")
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()
    run_blocking(args.split, args.dataset, args.cache_dir, args)

if __name__ == "__main__":
    main()
