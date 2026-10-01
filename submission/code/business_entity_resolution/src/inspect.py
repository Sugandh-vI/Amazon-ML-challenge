#!/usr/bin/env python3
"""inspect.py — the four measurement pre-checks agreed before try1 was built.

(A) country consistency across TRUE pairs (+ singleton rate / match
    cardinality per country)  -> decides whether country can be a blocking key
(B) full-canonicalization exact-match rates on a sample of true pairs
    (token-sorted name, suffix-stripped core, normalized address, house
    number, state)  -> sizes how much is reachable by exact keys
(C) Unicode script inventory of every source file (train + test)
(D) cross-script share among true pairs: pairs whose Source-2/3 name is
    non-Latin, non-Latin addresses, and the share of India S1 entities whose
    matches are ALL non-Latin (the structurally-hardest segment)

Stdlib + numpy only; streams the files, never holds one fully in memory.
"""

from __future__ import annotations

import os as _os, sys as _sys  # un-shadow stdlib 'inspect' (this dir has inspect.py)
_src_d = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0] or _os.getcwd()) == _src_d:
    _sys.path.pop(0)
    _sys.path.append(_src_d)


import argparse
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

import normalize as N

def log(msg: str) -> None:
    print(f"[inspect] {msg}", flush=True)

class Idx:
    """id -> (country, name_script, addr_nonascii) via sorted-array lookup."""

    def __init__(self):
        self._ids: List[int] = []
        self._cty: List[int] = []
        self._scr: List[int] = []
        self._ana: List[int] = []
        self.ids = None

    def add(self, i: int, cty: int, scr: int, ana: int) -> None:
        self._ids.append(i)
        self._cty.append(cty)
        self._scr.append(scr)
        self._ana.append(ana)

    def finalize(self) -> None:
        ids = np.asarray(self._ids, np.int64)
        order = np.argsort(ids, kind="stable")
        self.ids = ids[order]
        self.cty = np.asarray(self._cty, np.uint8)[order]
        self.scr = np.asarray(self._scr, np.uint8)[order]
        self.ana = np.asarray(self._ana, np.uint8)[order]
        self._ids = self._cty = self._scr = self._ana = []

    def lookup(self, q: np.ndarray, col: str) -> np.ndarray:
        pos = np.searchsorted(self.ids, q)
        pos = np.clip(pos, 0, len(self.ids) - 1)
        ok = self.ids[pos] == q
        out = np.zeros(len(q), np.uint8)
        out[ok] = getattr(self, col)[pos[ok]]
        return out

def hdr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74, flush=True)

def main():
    ap = argparse.ArgumentParser(description="try1 measurement pre-checks")
    N.add_common_args(ap)
    ap.add_argument("--sample", type=int, default=50_000,
                    help="true pairs sampled for checks B (default 50000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    gt_path = args.dataset / "train" / "train_ground_truth.tsv"
    if not gt_path.is_file():
        raise SystemExit(f"missing {gt_path}")

    # ---- GT arrays --------------------------------------------------------
    s1s, oth = [], []
    for s1, ids in N.stream_ground_truth(gt_path):
        for o in ids:
            s1s.append(s1)
            oth.append(o)
    gt_s1 = np.asarray(s1s, np.int64)
    gt_other = np.asarray(oth, np.int64)
    del s1s, oth
    order = np.lexsort((gt_other, gt_s1))
    gt_s1, gt_other = gt_s1[order], gt_other[order]
    log(f"GT pairs: {len(gt_s1):,}")

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(gt_s1), size=min(args.sample, len(gt_s1)),
                            replace=False)
    samp_s1 = gt_s1[sample_idx]
    samp_o = gt_other[sample_idx]
    # raw (file-space) ids: S3 codes carry the 2**31 offset
    samp_o_raw = np.where(samp_o >= N.S3_OFFSET, samp_o - N.S3_OFFSET, samp_o)
    capture_ids = set(int(x) for x in samp_s1) | set(int(x) for x in samp_o_raw)
    log(f"sampled {len(samp_s1):,} true pairs for check (B)")

    # ---- stream all six source files --------------------------------------
    idx = {"s1": Idx(), "s2": Idx(), "s3": Idx()}
    captured: Dict[int, Tuple[str, str, str]] = {}
    file_reports = []
    for split in ("train", "test"):
        for i in (1, 2, 3):
            path = args.dataset / split / f"{split}_source{i}.tsv"
            key = f"s{i}"
            scr_counts = Counter()
            n_rows = nonascii_addr = mojibake_ish = 0
            examples: List[str] = []
            t_file = time.time()
            for row in N.stream_source(path, limit=args.limit):
                n_rows += 1
                scr = N.detect_script(row.name)
                scr_counts[scr] += 1
                is_na_name = int(not row.name.isascii())
                is_na_addr = int(not row.addr.isascii())
                if is_na_addr:
                    nonascii_addr += 1
                if is_na_name and scr == "latin":
                    mojibake_ish += 1
                if scr != "latin" and len(examples) < 2:
                    examples.append(f"{row.prefix}-{row.id_num}: "
                                    f"{row.name[:60]!r}")
                if split == "train":
                    idx[key].add(
                        row.id_num,
                        N.COUNTRY_CODES.get(row.country.strip(), 255),
                        N.SCRIPT_CODES.get(scr, 255),
                        is_na_addr)
                    if row.id_num in capture_ids:
                        captured[row.id_num] = (row.name, row.addr,
                                                row.country)
            file_reports.append((path.name, n_rows, scr_counts,
                                 nonascii_addr, mojibake_ish, examples))
            log(f"scanned {path.name}: {n_rows:,} rows "
                f"({time.time()-t_file:.0f}s)")
    for ix in idx.values():
        ix.finalize()

    # =======================================================================
    hdr("(A) COUNTRY CONSISTENCY ON TRUE PAIRS + PER-COUNTRY CARDINALITY")
    c1 = idx["s1"].lookup(gt_s1, "cty")
    m2 = gt_other < N.S3_OFFSET
    m3 = ~m2
    co = np.zeros(len(gt_other), np.uint8)
    if m2.any():
        co[m2] = idx["s2"].lookup(gt_other[m2], "cty")
    if m3.any():
        co[m3] = idx["s3"].lookup(gt_other[m3] - N.S3_OFFSET, "cty")
    same = c1 == co
    n_unknown = int(((c1 == 255) | (co == 255)).sum())
    print(f"true pairs:                    {len(gt_s1):,}")
    print(f"same country label:            {int(same.sum()):,} "
          f"({100.0 * same.mean():.5f}%)")
    print(f"country MISMATCH:              {int((~same).sum()):,} "
          f"({100.0 * (~same).mean():.5f}%)   "
          f"[unknown-label pairs: {n_unknown}]")
    bad = np.flatnonzero(~same)[:10]
    for b in bad:
        print(f"  e.g. S1-{int(gt_s1[b])} ({N.COUNTRY_LABELS.get(int(c1[b]), c1[b])})"
              f"  <-> {N.decode_other(int(gt_other[b]))} "
              f"({N.COUNTRY_LABELS.get(int(co[b]), co[b])})")
    verdict = ("COUNTRY CAN BE A BLOCKING-KEY PREFIX (100% consistent)"
               if (~same).sum() == 0 else
               "country NOT perfectly consistent — keep as feature only, "
               "or key on it only for the consistent majority")
    print(f"VERDICT: {verdict}")

    # per-country cardinality / singletons (entity level over ALL S1 entities)
    uniq_s1, counts_per = np.unique(gt_s1, return_counts=True)
    ent_ids = idx["s1"].ids                       # every train S1 entity
    counts_all = np.zeros(len(ent_ids), np.int64)
    pos = np.searchsorted(ent_ids, uniq_s1)
    pos_c = np.clip(pos, 0, len(ent_ids) - 1)
    okm = ent_ids[pos_c] == uniq_s1
    counts_all[pos_c[okm]] = counts_per[okm]
    ent_cty = idx["s1"].lookup(ent_ids, "cty")
    print("\nper-country entity stats (train):")
    for code, label in N.COUNTRY_LABELS.items():
        sel = ent_cty == code
        if not sel.any():
            continue
        cc = counts_all[sel]
        print(f"  {label:8s} n={int(sel.sum()):>9,}  "
              f"singleton%={100.0 * (cc == 0).mean():5.1f}  "
              f"mean_matches={cc.mean():.3f}  median={int(np.median(cc))}  "
              f"max={int(cc.max())}")

    # =======================================================================
    hdr("(B) FULL-CANONICALIZATION EXACT-MATCH RATES ON TRUE PAIRS (sampled)")
    missing = sum(1 for i in samp_s1 if int(i) not in captured) + \
        sum(1 for i in samp_o_raw if int(i) not in captured)
    if missing:
        print(f"NOTE: {missing} sampled sides not captured"
              f"{' (use full run, no --limit)' if args.limit else ''}")
    rows: Dict[int, tuple] = {}
    for i in capture_ids:
        raw = captured.get(i)
        if raw is None:
            continue
        rn = N.normalize_row(i, raw[0], raw[1], raw[2])
        rows[i] = rn
    stats: Dict[str, Counter] = {"ALL": Counter()}
    n_eval = 0
    n_skip = 0
    for k in range(len(samp_s1)):
        a = rows.get(int(samp_s1[k]))
        b = rows.get(int(samp_o_raw[k]))
        if a is None or b is None:
            n_skip += 1
            continue
        n_eval += 1
        tag = N.COUNTRY_LABELS.get(a.country, str(a.country))
        s = stats.setdefault(tag, Counter())
        for field, denom in (("name_sorted", a.name_sorted != "" and b.name_sorted != ""),
                             ("name_core", a.name_core != "" and b.name_core != ""),
                             ("addr_norm", a.addr_norm != "" and b.addr_norm != ""),
                             ):
            if denom:
                s[field + "_den"] += 1
                stats["ALL"][field + "_den"] += 1
                eq = getattr(a, field) == getattr(b, field)
                if eq:
                    s[field + "_eq"] += 1
                    stats["ALL"][field + "_eq"] += 1
        if a.houseno and b.houseno:
            s["hn_den"] += 1
            stats["ALL"]["hn_den"] += 1
            if a.houseno == b.houseno:
                s["hn_eq"] += 1
                stats["ALL"]["hn_eq"] += 1
        if a.state and b.state:
            s["st_den"] += 1
            stats["ALL"]["st_den"] += 1
            if a.state == b.state:
                s["st_eq"] += 1
                stats["ALL"]["st_eq"] += 1
    if n_skip:
        print(f"(pairs skipped: {n_skip})")

    def pr(d, k_eq, k_den):
        den = d.get(k_den, 0)
        return f"{100.0 * d.get(k_eq, 0) / den:6.2f}%  ({d.get(k_eq,0):,}/{den:,})" if den else "   n/a"

    print(f"{'segment':10s} {'name_sorted=':>24s} {'name_core=':>24s} "
          f"{'addr_norm=':>24s}")
    for tag in sorted(stats, key=lambda t: (t != "ALL", t)):
        if tag == "ALL" or stats[tag].get("name_sorted_den", 0) > 0:
            d = stats[tag]
            print(f"{tag:10s} {pr(d, 'name_sorted_eq', 'name_sorted_den'):>24s} "
                  f"{pr(d, 'name_core_eq', 'name_core_den'):>24s} "
                  f"{pr(d, 'addr_norm_eq', 'addr_norm_den'):>24s}")
    print(f"\nhouse-number equal (both present): {pr(stats['ALL'], 'hn_eq', 'hn_den')}")
    print(f"state equal (both present):        {pr(stats['ALL'], 'st_eq', 'st_den')}")
    print("comparison keys: de-accented + casefolded + punctuation-stripped +")
    print("legal-suffix-stripped, token-sorted (i.e. exactly what the 'sig'/'pfx'")
    print("blocking keys use). Higher = more true pairs reachable by exact keys.")

    # =======================================================================
    hdr("(C) UNICODE SCRIPT INVENTORY PER FILE")
    print(f"{'file':22s} {'rows':>11s}  script mix (name)  | addr-nonascii  "
          f"| nonascii-name-but-latin (mojibake/accent)")
    for name, n_rows, scr, na_addr, moji, examples in file_reports:
        total = max(n_rows, 1)
        parts = ", ".join(
            f"{s}={100.0*c/total:.1f}%"
            for s, c in scr.most_common() if s != "latin" or c == scr["latin"])
        print(f"{name:22s} {n_rows:>11,}  {parts}")
        print(f"{'':22s} addr-nonascii={100.0*na_addr/total:.1f}%   "
              f"nonascii-name-but-latin={100.0*moji/total:.2f}%")
        for e in examples:
            print(f"{'':24s} e.g. {e}")
    print("\nNOTE: latin includes French accents (Latin-1); 'nonascii-name-but-latin'")
    print("      = accented-individual-letters (noise like 'Ínc') or mojibake.")

    # =======================================================================
    hdr("(D) CROSS-SCRIPT SHARE AMONG TRUE PAIRS")
    os_s = np.zeros(len(gt_other), np.uint8)
    os_a = np.zeros(len(gt_other), np.uint8)
    if m2.any():
        os_s[m2] = idx["s2"].lookup(gt_other[m2], "scr")
        os_a[m2] = idx["s2"].lookup(gt_other[m2], "ana")
    if m3.any():
        os_s[m3] = idx["s3"].lookup(gt_other[m3] - N.S3_OFFSET, "scr")
        os_a[m3] = idx["s3"].lookup(gt_other[m3] - N.S3_OFFSET, "ana")
    s1_scr = idx["s1"].lookup(gt_s1, "scr")
    nonlat_o = os_s != 0
    nonlat_1 = s1_scr != 0
    print(f"pairs with S1 name non-Latin:        {int(nonlat_1.sum()):,} "
          f"({100.0*nonlat_1.mean():.3f}%)  <- should be ~0 (S1 is Latin)")
    print(f"pairs with counterpart name non-Latin: {int(nonlat_o.sum()):,} "
          f"({100.0*nonlat_o.mean():.3f}%)")
    print(f"pairs with counterpart address non-ASCII: {int(os_a.sum()):,} "
          f"({100.0*(os_a>0).mean():.3f}%)")
    for code, label in ((1, "India"), (0, "US")):
        sel = c1 == code
        if sel.any():
            print(f"  [{label}] counterpart name non-Latin: "
                  f"{100.0*nonlat_o[sel].mean():.3f}% of pairs "
                  f"({int(sel.sum()):,} pairs)")
    # entity level: India entities whose ALL matches are non-Latin named
    matched_mask = np.ones(len(uniq_s1), bool)
    counts_lookup = counts_per > 0
    nonlat_pair_sorted = nonlat_o  # gt arrays sorted by s1 already
    starts = np.flatnonzero(np.diff(gt_s1)) + 1
    starts = np.concatenate(([0], starts, [len(gt_s1)]))
    all_nonlat = np.zeros(len(uniq_s1), bool)
    any_ent = np.zeros(len(uniq_s1), bool)
    pos_map = {int(e): i for i, e in enumerate(uniq_s1)}
    for a, b in zip(starts[:-1], starts[1:]):
        j = pos_map.get(int(gt_s1[a]))
        if j is None:
            continue
        any_ent[j] = True
        all_nonlat[j] = bool(nonlat_pair_sorted[a:b].all())
    india_sel = (idx["s1"].lookup(uniq_s1, "cty") == 1) & any_ent
    if india_sel.any():
        print(f"\nIndia entities with >=1 match:        "
              f"{int(india_sel.sum()):,}")
        print(f"  of which ALL matches non-Latin name: "
              f"{int((all_nonlat & india_sel).sum()):,} "
              f"({100.0*all_nonlat[india_sel].mean():.3f}%)")
        print("  -> these are structurally unreachable WITHOUT transliteration")
    all_sel = any_ent
    print(f"ALL entities, ALL matches non-Latin:   "
          f"{int((all_nonlat & all_sel).sum()):,} "
          f"({100.0*all_nonlat[all_sel].mean():.3f}% of matched entities)")

    print(f"\n[inspect] done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
