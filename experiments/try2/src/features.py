#!/usr/bin/env python3
"""features.py — pair feature computation for try1.

Two entry points used by both train.py and predict.py:

* :func:`compute_cheap_scores` — max(token_jaccard, token_containment) over
  token-sorted core names.  Used by blocking stage B and for the per-entity
  aggregate features.
* :func:`compute_features` — the full feature matrix (float32) for a batch of
  (S1, S2/S3) pairs, gathered from the normalization caches in bounded chunks
  so peak RAM stays ~chunk size.

Feature conventions
-------------------
* All similarities are in [0, 1].
* "missing" components are represented by explicit presence/xor flags, never
  by a 0 similarity (so 'both missing' and 'both different' stay separable).
* Entity-level aggregates (e_max/e_gap/e_mean5/e_n75) are functions of the
  cheap score over the entity's FULL candidate list (from blocking); they are
  looked up per row via :func:`aggs_for_rows`.
"""

from __future__ import annotations

import os as _os, sys as _sys  # un-shadow stdlib 'inspect' (this dir has inspect.py)
_src_d = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0] or _os.getcwd()) == _src_d:
    _sys.path.pop(0)
    _sys.path.append(_src_d)


from typing import Dict, List, Tuple

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Indel

import normalize as N
from normalize import S3_OFFSET

# Index map — keep in sync with FEATURE_NAMES.
#  0 jw_sorted      1 tsr_core      2 pr_sorted    3 ratio_sorted
#  4 indel_sorted   5 jw_translit   6 tsr_addr     7 indel_addr
#  8 pr_addr        9 jacc_core    10 contain_core 11 first_eq
# 12 last_eq       13 phon_eq      14 ini_eq      15 suff_jacc
# 16 ntok_diff     17 nchar_diff   18 addr_jacc   19 addr_contain
# 20 houseno_both  21 houseno_eq   22 houseno_xor 23 postal_both
# 24 postal_eq     25 postal_xor   26 state_both  27 state_eq
# 28 state_xor     29 name_in_addr 30 addr_in_name 31 prod_name_addr
# 32 min_name_addr 33 other_nonascii 34 cross_script 35 suff_count_diff
# 36 e_max  37 e_gap  38 e_mean5  39 e_n75
FEATURE_NAMES: List[str] = [
    "jw_sorted", "tsr_core", "pr_sorted", "ratio_sorted", "indel_sorted",
    "jw_translit", "tsr_addr", "indel_addr", "pr_addr",
    "jacc_core", "contain_core", "first_eq", "last_eq", "phon_eq", "ini_eq",
    "suff_jacc", "ntok_diff", "nchar_diff",
    "addr_jacc", "addr_contain",
    "houseno_both", "houseno_eq", "houseno_xor",
    "postal_both", "postal_eq", "postal_xor",
    "state_both", "state_eq", "state_xor",
    "name_in_addr", "addr_in_name", "prod_name_addr", "min_name_addr",
    "other_nonascii", "cross_script", "suff_count_diff",
    "e_max", "e_gap", "e_mean5", "e_n75",
]
N_FEATURES = len(FEATURE_NAMES)          # 40
AGG_COLS = (36, 37, 38, 39)
CHUNK = 500_000
_STR_COLS = ("name_sorted", "name_core", "name_suffixes", "name_initials",
             "name_phon", "name_translit", "addr_norm", "addr_tokens",
             "houseno", "postal", "state")
_NUM_COLS = ("name_script", "name_nonascii")

def feature_names() -> List[str]:
    return list(FEATURE_NAMES)

# --------------------------------------------------------------------------
def _other_rows(other_ents: np.ndarray, c2: N.SourceCache,
                c3: N.SourceCache) -> Tuple[np.ndarray, np.ndarray]:
    """S2/S3 entity codes -> (row indices valid for their own cache, is_s3)."""
    is_s3 = other_ents >= S3_OFFSET
    rows = np.zeros(len(other_ents), np.int64)
    m2 = ~is_s3
    m3 = is_s3
    if m2.any():
        rows[m2] = c2.row_of(other_ents[m2])
    if m3.any():
        rows[m3] = c3.row_of(other_ents[m3] - S3_OFFSET)
    return rows, is_s3

def _gather_str(cache: N.SourceCache, col: str, rows: np.ndarray) -> np.ndarray:
    return N.decode_str(cache.arrays[col][rows])

def compute_cheap_scores(cache1: N.SourceCache, c2: N.SourceCache,
                         c3: N.SourceCache, s1_ents: np.ndarray,
                         other_ents: np.ndarray) -> np.ndarray:
    """max(token_jaccard, token_containment) on name_sorted token sets."""
    n = len(s1_ents)
    out = np.zeros(n, np.float32)
    if n == 0:
        return out
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        r1 = cache1.row_of(s1_ents[lo:hi])
        r2, is3 = _other_rows(other_ents[lo:hi], c2, c3)
        r1s = np.maximum(r1, 0)
        r2s = np.maximum(r2, 0)
        a = _gather_str(cache1, "name_sorted", r1s)
        # gather each side with clamped rows, then select by side flag
        b2 = _gather_str(c2, "name_sorted", np.where(is3, 0, r2s))
        b3 = _gather_str(c3, "name_sorted", np.where(is3, r2s, 0))
        b = np.where(is3, b3, b2)
        a = np.where(r1 < 0, "", a)
        b = np.where(r2 < 0, "", b)
        for i in range(hi - lo):
            ta = a[i].split()
            tb = b[i].split()
            if not ta or not tb:
                continue
            sa, sb = set(ta), set(tb)
            inter = len(sa & sb)
            out[lo + i] = max(inter / float(len(sa | sb)),
                              inter / float(min(len(sa), len(sb))))
    return out

def _jac_contain(x: str, y: str):
    """(jaccard, containment) over whitespace token sets; (0,0) if either empty."""
    ta, tb = x.split(), y.split()
    if not ta or not tb:
        return 0.0, 0.0
    sa, sb = set(ta), set(tb)
    inter = len(sa & sb)
    if inter == 0:
        return 0.0, 0.0
    return (inter / float(len(sa | sb)),
            inter / float(min(len(sa), len(sb))))


def compute_cheap_scores_v2(cache1: N.SourceCache, c2: N.SourceCache,
                            c3: N.SourceCache, s1_ents: np.ndarray,
                            other_ents: np.ndarray):
    """Lexicographic ranker inputs -> (primary, secondary) float32.

    primary   = max(jaccard(name_sorted), jaccard(name_translit)) where the
                counterpart row is non-Latin (native script never overlaps
                Latin tokens, so without this cross-script pairs score 0).
    secondary = containment, same max-with-translit treatment — used ONLY to
                break primary ties (Phase 0 showed max(jaccard, containment)
                lets subset-names saturate at 1.0 and bury true pairs among
                ties).
    """
    n = len(s1_ents)
    pri = np.zeros(n, np.float32)
    sec = np.zeros(n, np.float32)
    if n == 0:
        return pri, sec
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        r1 = cache1.row_of(s1_ents[lo:hi])
        r2, is3 = _other_rows(other_ents[lo:hi], c2, c3)
        r1s = np.maximum(r1, 0)
        r2s = np.maximum(r2, 0)
        a = _gather_str(cache1, "name_sorted", r1s)
        b2 = _gather_str(c2, "name_sorted", np.where(is3, 0, r2s))
        b3 = _gather_str(c3, "name_sorted", np.where(is3, r2s, 0))
        b = np.where(is3, b3, b2)
        a = np.where(r1 < 0, "", a)
        b = np.where(r2 < 0, "", b)
        sc2 = c2.arrays["name_script"][np.where(is3, 0, r2s)]
        sc3 = c3.arrays["name_script"][np.where(is3, r2s, 0)]
        sc = np.where(is3, sc3, sc2)
        sc = np.where(r2 < 0, 0, sc)
        nonlat = sc != 0
        if nonlat.any():
            at = _gather_str(cache1, "name_translit", r1s)
            bt2 = _gather_str(c2, "name_translit", np.where(is3, 0, r2s))
            bt3 = _gather_str(c3, "name_translit", np.where(is3, r2s, 0))
            bt = np.where(is3, bt3, bt2)
            at = np.where(r1 < 0, "", at)
            bt = np.where(r2 < 0, "", bt)
        for i in range(hi - lo):
            j, c = _jac_contain(a[i], b[i])
            if nonlat[i]:
                jt, ct = _jac_contain(at[i], bt[i])
                if jt > j:
                    j = jt
                if ct > c:
                    c = ct
            pri[lo + i] = j
            sec[lo + i] = c
    return pri, sec


def entity_aggs(s1_ids: np.ndarray, indptr: np.ndarray,
                cheap: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-candidate-list aggregates from cheap scores (CSR-aligned arrays)."""
    n = len(s1_ids)
    e_max = np.zeros(n, np.float32)
    e_gap = np.zeros(n, np.float32)
    e_mean5 = np.zeros(n, np.float32)
    e_n75 = np.zeros(n, np.float32)
    for g in range(n):
        seg = cheap[indptr[g]:indptr[g + 1]]
        if len(seg) == 0:
            continue
        s = np.sort(seg)[::-1]
        e_max[g] = s[0]
        e_gap[g] = s[0] - s[1] if len(s) > 1 else float(s[0])
        e_mean5[g] = float(s[:5].mean())
        e_n75[g] = float((seg >= 0.75).sum())
    return {"e_max": e_max, "e_gap": e_gap, "e_mean5": e_mean5,
            "e_n75": e_n75}

def aggs_for_rows(tables: Dict[str, np.ndarray], csr_s1: np.ndarray,
                  row_s1_ents: np.ndarray) -> np.ndarray:
    """Look up CSR-aligned entity aggregates for arbitrary S1 entities.

    Entities without candidates (absent from the CSR) get zeros.
    """
    out = np.zeros((len(row_s1_ents), 4), np.float32)
    if len(csr_s1) == 0:
        return out
    pos = np.searchsorted(csr_s1, row_s1_ents)
    pos_c = np.clip(pos, 0, len(csr_s1) - 1)
    ok = csr_s1[pos_c] == row_s1_ents
    if ok.any():
        for j, key in enumerate(("e_max", "e_gap", "e_mean5", "e_n75")):
            out[ok, j] = tables[key][pos_c[ok]]
    return out

# --------------------------------------------------------------------------
def compute_features(cache1: N.SourceCache, c2: N.SourceCache,
                     c3: N.SourceCache, s1_ents: np.ndarray,
                     other_ents: np.ndarray, agg_rows: np.ndarray,
                     out: np.ndarray = None) -> np.ndarray:
    """Full feature matrix for pairs (s1_ents, other_ents).

    ``agg_rows`` is an (N, 4) matrix of entity aggregates (from
    :func:`aggs_for_rows`), appended as the last four columns.  ``out`` may be
    a preallocated (or memmapped) (N, N_FEATURES) float32 array.
    """
    n = len(s1_ents)
    if out is None:
        out = np.zeros((n, N_FEATURES), np.float32)
    if agg_rows.shape != (n, 4):
        raise ValueError(f"agg_rows shape {agg_rows.shape} != ({n}, 4)")
    out[:, AGG_COLS] = agg_rows
    if n == 0:
        return out

    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        sl = slice(lo, hi)
        r1 = cache1.row_of(s1_ents[sl])
        r2, is3 = _other_rows(other_ents[sl], c2, c3)
        if (r1 < 0).any() or (r2 < 0).any():
            raise ValueError(
                "pair references an entity id absent from cache "
                f"(s1 misses={int((r1 < 0).sum())}, "
                f"other misses={int((r2 < 0).sum())})")
        r1s, r2s = np.maximum(r1, 0), np.maximum(r2, 0)
        r2_s2 = np.where(is3, 0, r2s)
        r2_s3 = np.where(is3, r2s, 0)

        g1 = {c: _gather_str(cache1, c, r1s) for c in _STR_COLS}
        g1s = {c: np.asarray(cache1.arrays[c][r1s]).astype(np.int16)
               for c in _NUM_COLS}
        # other side: gather from both caches with clamped rows, select later
        go_str = {c: None for c in _STR_COLS}
        for c in _STR_COLS:
            v2 = _gather_str(c2, c, r2_s2)
            v3 = _gather_str(c3, c, r2_s3)
            go_str[c] = np.where(is3, v3, v2)
        go_num = {}
        for c in _NUM_COLS:
            v2 = np.asarray(c2.arrays[c][r2_s2]).astype(np.int16)
            v3 = np.asarray(c3.arrays[c][r2_s3]).astype(np.int16)
            go_num[c] = np.where(is3, v3, v2)

        s1_sorted = g1["name_sorted"]
        o_sorted = go_str["name_sorted"]
        s1_core = g1["name_core"]
        o_core = go_str["name_core"]
        s1_addr = g1["addr_norm"]
        o_addr = go_str["addr_norm"]
        s1_atok = [frozenset(x.split()) for x in g1["addr_tokens"]]
        o_atok = [frozenset(x.split()) for x in go_str["addr_tokens"]]
        o_translit = go_str["name_translit"]

        m = hi - lo
        X = out[sl]
        for i in range(m):
            ns1, ns2 = s1_sorted[i], o_sorted[i]
            c1, c2t = s1_core[i], o_core[i]
            a1, a2 = s1_addr[i], o_addr[i]
            # --- fuzzy name features ------------------------------------
            X[i, 0] = JaroWinkler.normalized_similarity(ns1, ns2)
            X[i, 1] = fuzz.token_set_ratio(c1, c2t) / 100.0
            X[i, 2] = fuzz.partial_ratio(ns1, ns2) / 100.0
            X[i, 3] = fuzz.ratio(ns1, ns2) / 100.0
            X[i, 4] = Indel.normalized_similarity(ns1, ns2)
            X[i, 5] = JaroWinkler.normalized_similarity(ns1, o_translit[i])
            X[i, 6] = fuzz.token_set_ratio(a1, a2) / 100.0
            X[i, 7] = Indel.normalized_similarity(a1, a2)
            X[i, 8] = fuzz.partial_ratio(a1, a2) / 100.0
            # --- set-based name features --------------------------------
            t1 = c1.split()
            t2 = c2t.split()
            if t1 and t2:
                sa, sb = set(t1), set(t2)
                inter = len(sa & sb)
                union = len(sa | sb)
                X[i, 9] = inter / float(union)
                X[i, 10] = inter / float(min(len(sa), len(sb)))
                X[i, 11] = float(t1[0] == t2[0])
                X[i, 12] = float(t1[-1] == t2[-1])
                X[i, 16] = abs(len(t1) - len(t2)) / float(max(len(t1), len(t2)))
            X[i, 13] = float(bool(g1["name_phon"][i])
                             and g1["name_phon"][i] == go_str["name_phon"][i])
            ini1 = g1["name_initials"][i]
            X[i, 14] = float(len(ini1) >= 3 and ini1 == go_str["name_initials"][i])
            su1 = set(g1["name_suffixes"][i].split())
            su2 = set(go_str["name_suffixes"][i].split())
            if not su1 and not su2:
                X[i, 15] = 1.0
            elif su1 or su2:
                X[i, 15] = len(su1 & su2) / float(len(su1 | su2))
            X[i, 17] = abs(len(ns1) - len(ns2)) / float(max(len(ns1), len(ns2), 1))
            # --- address features ---------------------------------------
            fa, fb = s1_atok[i], o_atok[i]
            if fa and fb:
                inter_a = len(fa & fb)
                X[i, 18] = inter_a / float(len(fa | fb))
                X[i, 19] = inter_a / float(min(len(fa), len(fb)))
            # house number / postal / state with explicit presence flags
            for base, key in ((20, "houseno"), (23, "postal"), (26, "state")):
                v1 = g1[key][i]
                v2 = go_str[key][i]
                p1, p2 = bool(v1), bool(v2)
                X[i, base] = float(p1 and p2)              # both present
                X[i, base + 1] = float(p1 and p2 and v1 == v2)  # equal
                X[i, base + 2] = float(p1 != p2)           # one missing
            # name <-> address overlap
            if t1:
                X[i, 29] = sum(1 for t in t1 if t in o_atok[i]) / float(len(t1))
            if t2:
                X[i, 30] = sum(1 for t in t2 if t in s1_atok[i]) / float(len(t2))
            X[i, 31] = X[i, 9] * X[i, 18]
            X[i, 32] = min(X[i, 9], X[i, 18])
            # --- script / misc ------------------------------------------
            X[i, 33] = float(go_num["name_nonascii"][i])
            X[i, 34] = float(g1s["name_script"][i] != go_num["name_script"][i])
            X[i, 35] = abs(len(su1) - len(su2))
    return out
