#!/usr/bin/env python3
"""train.py — pair dataset assembly, LightGBM fit, validation scoring.

Run after ``blocking.py --split train``.  Produces:

* model/lgbm.txt             trained booster (LightGBM text format)
* model/train_meta.json      feature list, hyperparameters, split info
* cache/split_val_s1.npy     validation S1 entity ids (entity-level split)
* cache/val_scores.npz       (s1, other, raw score) for every validation
                             candidate — consumed by tune.py
* cache/val_gt.npz           ground-truth pairs of validation entities

Sampling design
---------------
* learn rows    = ALL positives of learn-entities + up to --neg-cap hard
                  negatives per entity (drawn from that entity's blocking
                  candidates, i.e. the exact inference-time confusers)
* stop rows     = ALL candidates of early-stop entities (held out from
                  fitting; used for AUC + early stopping)
* val rows      = ALL candidates of validation entities (threshold tuning
                  in tune.py happens on this split — no label leakage)

The split is entity-level and stratified by (country x singleton x cardinality)
so no S1 entity leaks across learn/stop/val.
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
from typing import Dict, List, Tuple

import numpy as np

import normalize as N
import features as F
from blocking import load_candidates, pack_pairs

def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)

# --------------------------------------------------------------------------
def load_gt_pairs(gt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """All GT pairs as sorted-by-s1 (s1, other) int64 arrays."""
    counts_s1: List[int] = []
    counts_n: List[int] = []
    total = 0
    for s1, ids in N.stream_ground_truth(gt_path):
        counts_s1.append(s1)
        counts_n.append(len(ids))
        total += len(ids)
    s1_arr = np.empty(total, np.int64)
    o_arr = np.empty(total, np.int64)
    pos = 0
    for s1, ids in N.stream_ground_truth(gt_path):
        for o in ids:
            s1_arr[pos] = s1
            o_arr[pos] = o
            pos += 1
    order = np.lexsort((o_arr, s1_arr))
    return s1_arr[order], o_arr[order]

def gt_group_bounds(s1_sorted: np.ndarray, ents: np.ndarray):
    """(lo, hi) index bounds into a sorted s1 array for each entity id."""
    lo = np.searchsorted(s1_sorted, ents, side="left")
    hi = np.searchsorted(s1_sorted, ents, side="right")
    return lo, hi

def stratified_split(s1_all: np.ndarray, counts: np.ndarray,
                     country: np.ndarray, val_frac: float, seed: int):
    bucket = np.select(
        [counts == 0, counts == 1, counts == 2, counts <= 4],
        [0, 1, 2, 3], default=4).astype(np.int64)
    strata = country.astype(np.int64) * 16 + bucket
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(s1_all), dtype=bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        take = max(1, int(round(val_frac * len(idx))))
        val_mask[idx[:take]] = True
    val_ids = np.sort(s1_all[val_mask])
    fit_ids = np.sort(s1_all[~val_mask])
    return fit_ids, val_ids

# --------------------------------------------------------------------------
def entity_loop(csr_s1, indptr, others, entities, gt_s1, gt_other,
                gt_lo, gt_hi, neg_cap, seed, want_negs: bool,
                want_all_cand: bool):
    """Loop over given entities; sample negatives / collect candidates.

    Returns (s1_rows, other_rows) pair arrays for the requested entities.
    """
    rng = np.random.default_rng(seed)
    s1_parts: List[np.ndarray] = []
    o_parts: List[np.ndarray] = []
    pos_in_csr = np.searchsorted(csr_s1, entities)
    has = (pos_in_csr < len(csr_s1)) & (csr_s1[
        np.clip(pos_in_csr, 0, len(csr_s1) - 1)] == entities)
    pos_c = np.clip(pos_in_csr, 0, len(csr_s1) - 1)
    for i, ent in enumerate(entities):
        # positives (only for learn mode — eval modes score candidate rows)
        lo, hi = int(gt_lo[i]), int(gt_hi[i])
        if not want_all_cand and hi > lo:
            s1_parts.append(gt_s1[lo:hi])
            o_parts.append(gt_other[lo:hi])
        # negatives / all candidates
        if want_all_cand or want_negs:
            if has[i]:
                cl, ch = int(indptr[pos_c[i]]), int(indptr[pos_c[i] + 1])
                cs = others[cl:ch]
                if want_all_cand:
                    s1_parts.append(np.full(len(cs), ent, np.int64))
                    o_parts.append(cs)
                elif len(cs) and neg_cap > 0:
                    gt_set = set(int(x) for x in gt_other[lo:hi])
                    # candidates that are NOT true matches
                    mask = np.fromiter(
                        (int(x) not in gt_set for x in cs),
                        dtype=bool, count=len(cs))
                    negs = cs[mask]
                    if len(negs) > neg_cap:
                        negs = rng.choice(negs, size=neg_cap, replace=False)
                    if len(negs):
                        s1_parts.append(np.full(len(negs), ent, np.int64))
                        o_parts.append(negs)
    if not s1_parts:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(s1_parts), np.concatenate(o_parts)

def build_X(cache1, c2, c3, s1_ents, other_ents, tables, path: Path) -> None:
    """Write the feature matrix for pairs to a float32 .npy memmap at path."""
    n = len(s1_ents)
    agg = F.aggs_for_rows(tables, tables["_csr_s1"], s1_ents)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                   shape=(n, F.N_FEATURES))
    t0 = time.time()
    F.compute_features(cache1, c2, c3, s1_ents, other_ents, agg, out=mm)
    mm.flush()
    del mm
    log(f"features[{path.name}]: {n:,} rows in {time.time()-t0:.0f}s")

def main():
    ap = argparse.ArgumentParser(description="try1 model training")
    N.add_common_args(ap)
    ap.add_argument("--neg-cap", type=int, default=6,
                    help="max sampled hard negatives per learn entity")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--stop-frac", type=float, default=0.15,
                    help="fraction of fit-side entities held out for early stop")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if val_scores/model exist")
    args = ap.parse_args()

    cache_dir: Path = args.cache_dir
    model_dir = N.TRY1_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    vs_path = cache_dir / "val_scores.npz"
    if vs_path.is_file() and (model_dir / "lgbm.txt").is_file() and not args.force:
        log("outputs already exist (use --force to redo) — skipping to the end")
        return

    t_all = time.time()
    gt_path = args.dataset / "train" / "train_ground_truth.tsv"
    if not gt_path.is_file():
        raise SystemExit(f"missing {gt_path}")
    if not (cache_dir / "candidates_train.npz").is_file():
        raise SystemExit("run blocking.py --split train first "
                         "(candidates_train.npz not found)")

    caches = N.resolve_split_stems("train", args.dataset, cache_dir,
                                   limit=args.limit)
    log(f"caches loaded ({time.time()-t_all:.0f}s)")

    # ---- GT ---------------------------------------------------------------
    gt_s1, gt_other = load_gt_pairs(gt_path)
    log(f"GT pairs: {len(gt_s1):,}")
    uniq_s1, gt_counts_all = np.unique(gt_s1, return_counts=True)
    s1_all = caches[1].ids.astype(np.int64)
    counts = np.zeros(len(s1_all), np.int64)
    pos_idx = np.searchsorted(uniq_s1, s1_all)
    pos_c = np.clip(pos_idx, 0, len(uniq_s1) - 1)
    ok = uniq_s1[pos_c] == s1_all
    counts[ok] = gt_counts_all[pos_c[ok]]
    country = caches[1].get("country", np.arange(caches[1].rows)).astype(np.int64)

    # ---- entity-level stratified split ------------------------------------
    fit_ids, val_ids = stratified_split(s1_all, counts, country,
                                         args.val_frac, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(len(fit_ids))
    n_stop = max(1, int(round(args.stop_frac * len(fit_ids))))
    stop_ids = np.sort(fit_ids[perm[:n_stop]])
    learn_ids = np.sort(fit_ids[perm[n_stop:]])
    np.save(cache_dir / "split_val_s1.npy", val_ids)
    log(f"split: learn={len(learn_ids):,} stop={len(stop_ids):,} "
        f"val={len(val_ids):,} entities")

    # ---- candidates + per-entity cheap scores -> agg tables ---------------
    csr_s1, indptr, others = load_candidates(cache_dir, "train")
    log(f"candidates: {len(others):,} pairs over {len(csr_s1):,} S1")
    t0 = time.time()
    n_g = len(csr_s1)
    e_max = np.zeros(n_g, np.float32)
    e_gap = np.zeros(n_g, np.float32)
    e_mean5 = np.zeros(n_g, np.float32)
    e_n75 = np.zeros(n_g, np.float32)
    for g in range(n_g):
        lo, hi = int(indptr[g]), int(indptr[g + 1])
        ent = int(csr_s1[g])
        cheap = F.compute_cheap_scores(
            caches[1], caches[2], caches[3],
            np.full(hi - lo, ent, np.int64), others[lo:hi])
        s = np.sort(cheap)[::-1]
        if len(s):
            e_max[g] = s[0]
            e_gap[g] = s[0] - s[1] if len(s) > 1 else float(s[0])
            e_mean5[g] = float(s[:5].mean())
            e_n75[g] = float((cheap >= 0.75).sum())
        if g and g % 200_000 == 0:
            log(f"  cheap-score pass {g:,}/{n_g:,} "
                f"({time.time()-t0:.0f}s)")
    tables = {"e_max": e_max, "e_gap": e_gap, "e_mean5": e_mean5,
              "e_n75": e_n75, "_csr_s1": csr_s1}
    log(f"entity agg tables built ({time.time()-t0:.0f}s)")

    l_lo, l_hi = gt_group_bounds(gt_s1, learn_ids)
    s_lo, s_hi = gt_group_bounds(gt_s1, stop_ids)
    v_lo, v_hi = gt_group_bounds(gt_s1, val_ids)

    # ---- pair sets --------------------------------------------------------
    log("building learn pairs (pos + sampled hard negatives) ...")
    ls1, lo_ = entity_loop(csr_s1, indptr, others, learn_ids,
                           gt_s1, gt_other, l_lo, l_hi,
                           args.neg_cap, args.seed, want_negs=True,
                           want_all_cand=False)
    log("building stop pairs (all candidates) ...")
    ss1, so_ = entity_loop(csr_s1, indptr, others, stop_ids,
                           gt_s1, gt_other, s_lo, s_hi,
                           0, args.seed, want_negs=False, want_all_cand=True)
    log("building val pairs (all candidates) ...")
    vs1, vo_ = entity_loop(csr_s1, indptr, others, val_ids,
                           gt_s1, gt_other, v_lo, v_hi,
                           0, args.seed, want_negs=False, want_all_cand=True)

    # labels for learn rows: positive iff pair in GT
    gt_pack = pack_pairs(gt_s1, gt_other)
    lp = pack_pairs(ls1, lo_)
    y_learn = np.isin(lp, gt_pack).astype(np.uint8)
    del lp, gt_pack
    log(f"rows: learn={len(ls1):,} (pos={int(y_learn.sum()):,}) "
        f"stop={len(ss1):,} val={len(vs1):,} "
        f"({time.time()-t_all:.0f}s total so far)")

    # order rows by entity (contiguity helps debugging / nothing else)
    o1 = np.lexsort((lo_, ls1)); ls1, lo_, y_learn = ls1[o1], lo_[o1], y_learn[o1]
    o2 = np.lexsort((so_, ss1)); ss1, so_ = ss1[o2], so_[o2]
    o3 = np.lexsort((vo_, vs1)); vs1, vo_ = vs1[o3], vo_[o3]

    # ---- features ---------------------------------------------------------
    build_X(caches[1], caches[2], caches[3], ls1, lo_, tables,
            cache_dir / "X_learn.npy")
    build_X(caches[1], caches[2], caches[3], ss1, so_, tables,
            cache_dir / "X_stop.npy")
    build_X(caches[1], caches[2], caches[3], vs1, vo_, tables,
            cache_dir / "X_val.npy")
    np.save(cache_dir / "y_learn.npy", y_learn)
    np.save(cache_dir / "pairs_stop.npy", np.stack([ss1, so_]))
    np.save(cache_dir / "pairs_val.npy", np.stack([vs1, vo_]))

    # val GT for tune.py
    v_gt_lo, v_gt_hi = gt_group_bounds(gt_s1, val_ids)
    # GT may contain entities with zero pairs (singletons) — filter empties
    keep_e = v_gt_hi > v_gt_lo
    parts_s = [gt_s1[v_gt_lo[i]:v_gt_hi[i]] for i in np.flatnonzero(keep_e)]
    parts_o = [gt_other[v_gt_lo[i]:v_gt_hi[i]] for i in np.flatnonzero(keep_e)]
    val_gt_s1 = np.concatenate(parts_s) if parts_s else np.empty(0, np.int64)
    val_gt_other = np.concatenate(parts_o) if parts_o else np.empty(0, np.int64)
    np.savez(cache_dir / "val_gt.npz", s1=val_gt_s1, other=val_gt_other)

    # ---- LightGBM ---------------------------------------------------------
    from lightgbm import LGBMClassifier, early_stopping
    X_learn = np.load(cache_dir / "X_learn.npy", mmap_mode="r")
    X_stop = np.load(cache_dir / "X_stop.npy", mmap_mode="r")
    n = len(y_learn)
    y_stop = np.isin(pack_pairs(ss1, so_),
                     pack_pairs(gt_s1, gt_other)).astype(np.uint8)
    tiny = n < 2000
    params = dict(
        n_estimators=100 if tiny else 3000,
        learning_rate=0.05,
        num_leaves=15 if tiny else 127,
        max_depth=-1,
        min_child_samples=5 if tiny else 50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=args.seed,
    )
    model = LGBMClassifier(objective="binary", metric="binary_logloss", **params)
    if not tiny:
        model.fit(X_learn, y_learn,
                  eval_set=[(X_stop, y_stop)],
                  eval_metric="binary_logloss",
                  callbacks=[early_stopping(100, verbose=False)])
    else:
        model.fit(X_learn, y_learn)

    stop_proba = model.predict_proba(X_stop)[:, 1]
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_stop, stop_proba)
    except Exception:
        auc = float("nan")
    log(f"early-stop AUC: {auc:.5f}")

    model.booster_.save_model(str(model_dir / "lgbm.txt"))
    meta = {
        "feature_names": F.feature_names(),
        "params": {k: v for k, v in params.items()},
        "auc_stop": float(auc),
        "rows_learn": int(n), "pos_learn": int(y_learn.sum()),
        "rows_stop": int(len(ss1)), "rows_val": int(len(vs1)),
        "neg_cap": args.neg_cap, "seed": args.seed,
        "val_frac": args.val_frac, "stop_frac": args.stop_frac,
        "n_learn_entities": int(len(learn_ids)),
        "n_stop_entities": int(len(stop_ids)),
        "n_val_entities": int(len(val_ids)),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
    }
    (model_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- val scores -------------------------------------------------------
    X_val = np.load(cache_dir / "X_val.npy", mmap_mode="r")
    val_proba = model.predict_proba(X_val)[:, 1]
    np.savez(vs_path, s1=vs1, other=vo_, score=val_proba.astype(np.float32))
    log(f"val scores saved -> {vs_path}")

    # cleanup big intermediates (keep val pairs/gt, drop matrices)
    for f in ("X_learn.npy", "X_stop.npy", "X_val.npy", "y_learn.npy",
              "pairs_stop.npy", "pairs_val.npy"):
        p = cache_dir / f
        if p.is_file():
            p.unlink()
    log(f"done in {time.time()-t_all:.0f}s")

if __name__ == "__main__":
    main()
