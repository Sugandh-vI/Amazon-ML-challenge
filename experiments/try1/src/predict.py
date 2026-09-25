#!/usr/bin/env python3
"""predict.py — end-to-end inference on the test split.

Steps:
  1. build/verify the test normalization cache
  2. run blocking (skipped when cache/candidates_test.npz already exists,
     unless --force-blocking) -> writes output/candidate_pairs.tsv
  3. score every test candidate with the trained LightGBM model
  4. apply model/decision.json (pair_threshold + entity_gate)
  5. write output/matching_results.tsv (one row per test S1 entity)
  6. run the official validator (student_resource/utils/validate_submission.py)

The score loop is chunked: features for ~--chunk pairs are computed, scored,
and discarded — peak RAM stays bounded regardless of candidate volume.
"""

from __future__ import annotations

import os as _os, sys as _sys  # un-shadow stdlib 'inspect' (this dir has inspect.py)
_src_d = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0] or _os.getcwd()) == _src_d:
    _sys.path.pop(0)
    _sys.path.append(_src_d)


import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import normalize as N
import features as F
from blocking import load_candidates, run_blocking, write_candidate_tsv

def log(msg: str) -> None:
    print(f"[predict] {msg}", flush=True)

def build_agg_tables(caches, csr_s1, indptr, others, chunk_report=200_000):
    """Cheap-score pass over all candidates -> per-entity aggregate tables."""
    n_g = len(csr_s1)
    tables = {k: np.zeros(n_g, np.float32)
              for k in ("e_max", "e_gap", "e_mean5", "e_n75")}
    t0 = time.time()
    for g in range(n_g):
        lo, hi = int(indptr[g]), int(indptr[g + 1])
        ent = int(csr_s1[g])
        cheap = F.compute_cheap_scores(
            caches[1], caches[2], caches[3],
            np.full(hi - lo, ent, np.int64), others[lo:hi])
        s = np.sort(cheap)[::-1]
        if len(s):
            tables["e_max"][g] = s[0]
            tables["e_gap"][g] = s[0] - s[1] if len(s) > 1 else float(s[0])
            tables["e_mean5"][g] = float(s[:5].mean())
            tables["e_n75"][g] = float((cheap >= 0.75).sum())
        if g and g % chunk_report == 0:
            log(f"  agg pass {g:,}/{n_g:,} ({time.time()-t0:.0f}s)")
    log(f"agg tables: {n_g:,} entities ({time.time()-t0:.0f}s)")
    return tables

def score_all(caches, csr_s1, indptr, others, tables, booster,
              chunk: int) -> np.ndarray:
    """Raw model scores for every candidate (CSR-aligned)."""
    counts = np.diff(indptr) if len(csr_s1) else np.zeros(0, np.int64)
    s1_pairs = np.repeat(csr_s1, counts) if len(csr_s1) else np.empty(0, np.int64)
    n = len(s1_pairs)
    scores = np.zeros(n, np.float32)
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        agg = F.aggs_for_rows(tables, csr_s1, s1_pairs[lo:hi])
        X = F.compute_features(caches[1], caches[2], caches[3],
                               s1_pairs[lo:hi], others[lo:hi], agg)
        scores[lo:hi] = booster.predict(X).astype(np.float32)
        done = hi
        if lo and lo % (chunk * 10) < chunk:
            log(f"  scored {done:,}/{n:,} ({time.time()-t0:.0f}s)")
    log(f"scored {n:,} candidates in {time.time()-t0:.0f}s")
    return scores

def write_matching_tsv(path: Path, s1_all_ids: np.ndarray, csr_s1,
                       indptr, others, scores, tau, gate):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = n_rows = n_emitted = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for ent in s1_all_ids:
            n_rows += 1
            line = ""
            pos = int(np.searchsorted(csr_s1, ent))
            if pos < len(csr_s1) and csr_s1[pos] == ent:
                lo, hi = int(indptr[pos]), int(indptr[pos + 1])
                seg_scores = scores[lo:hi]
                keep = seg_scores >= tau
                if gate is not None and (len(seg_scores) == 0
                                         or seg_scores.max() < gate):
                    keep = np.zeros(len(seg_scores), dtype=bool)
                if keep.any():
                    line = ",".join(N.decode_other(int(c))
                                    for c in others[lo:hi][keep])
                    n_emitted += int(keep.sum())
            if not line:
                n_empty += 1
            f.write(f"S1-{int(ent)}\t{line}\n")
    log(f"wrote {path}: {n_rows:,} rows, {n_empty:,} empty "
        f"(predicted singletons), {n_emitted:,} matched ids total")

def main():
    ap = argparse.ArgumentParser(description="try1 end-to-end prediction")
    N.add_common_args(ap)
    ap.add_argument("--force-blocking", action="store_true")
    ap.add_argument("--chunk", type=int, default=1_000_000,
                    help="pairs scored per chunk (default %(default)s)")
    ap.add_argument("--k", type=int, default=None,
                    help="override stage-B cap when blocking runs")
    ap.add_argument("--per-key-cap", type=int, default=5000)
    ap.add_argument("--budget", type=int, default=200_000_000)
    ap.add_argument("--skip-validator", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    model_dir = N.TRY1_DIR / "model"
    booster_path = model_dir / "lgbm.txt"
    decision_path = model_dir / "decision.json"
    if not booster_path.is_file():
        raise SystemExit("model/lgbm.txt not found — run train.py first")
    if not decision_path.is_file():
        raise SystemExit("model/decision.json not found — run tune.py first")
    decision = json.loads(decision_path.read_text())
    tau = float(decision["pair_threshold"])
    gate = decision.get("entity_gate")
    gate = None if gate is None else float(gate)
    log(f"decision: pair_threshold={tau} entity_gate={gate}")

    # ---- 1-2: cache + blocking -------------------------------------------
    cand_npz = args.cache_dir / "candidates_test.npz"
    if args.force_blocking or not cand_npz.is_file():
        class _A:
            pass
        bargs = _A()
        bargs.limit = args.limit
        bargs.force_cache = args.force_blocking
        bargs.k = args.k if args.k is not None else 80
        bargs.per_key_cap = args.per_key_cap
        bargs.budget = args.budget
        run_blocking("test", args.dataset, args.cache_dir, bargs)
    else:
        log("reuse existing cache/candidates_test.npz (--force-blocking to redo)")

    caches = N.resolve_split_stems("test", args.dataset, args.cache_dir,
                                   limit=args.limit)
    csr_s1, indptr, others = load_candidates(args.cache_dir, "test")
    log(f"candidates: {len(others):,} pairs over {len(csr_s1):,} S1 entities")

    # candidate_pairs.tsv is the stage-B output — (re)write every run so the
    # file on disk always matches the npz being scored.
    write_candidate_tsv(N.TRY1_DIR / "output" / "candidate_pairs.tsv",
                        caches[1].ids, csr_s1, indptr, others)

    # ---- 3: score ---------------------------------------------------------
    from lightgbm import Booster
    booster = Booster(model_file=str(booster_path))
    meta = json.loads((model_dir / "train_meta.json").read_text())
    if meta["feature_names"] != F.feature_names():
        raise SystemExit("feature list mismatch between train_meta.json and "
                         "features.py — the model and code are out of sync")
    tables = build_agg_tables(caches, csr_s1, indptr, others)
    scores = score_all(caches, csr_s1, indptr, others, tables, booster,
                       args.chunk)

    # ---- 4-5: decision + output ------------------------------------------
    write_matching_tsv(N.TRY1_DIR / "output" / "matching_results.tsv",
                       caches[1].ids, csr_s1, indptr, others, scores, tau, gate)

    # ---- 6: validator -----------------------------------------------------
    if not args.skip_validator:
        validator = N.REPO_ROOT / "student_resource" / "utils" / \
            "validate_submission.py"
        cmd = [sys.executable, str(validator),
               "--matching", str(N.TRY1_DIR / "output" / "matching_results.tsv"),
               "--candidate", str(N.TRY1_DIR / "output" / "candidate_pairs.tsv"),
               "--test-dir", str(args.dataset / "test")]
        log("running official validator ...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            log("VALIDATOR FAILED (exit %d)" % res.returncode)
        else:
            log("validator PASS")
    log(f"total: {time.time()-t_all:.0f}s")

if __name__ == "__main__":
    main()
