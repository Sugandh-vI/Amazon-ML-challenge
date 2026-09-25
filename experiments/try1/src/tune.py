#!/usr/bin/env python3
"""tune.py — threshold + singleton-gate selection on the validation split.

Reads cache/val_scores.npz (raw LightGBM scores for every validation
candidate) + cache/val_gt.npz, then grid-searches:

    emit pair iff  score >= pair_threshold
    AND (entity has max score >= entity_gate, if a gate is configured)

directly maximising MACRO F_0.5 exactly as the leaderboard computes it:

    * per S1 entity:  empty prediction scores 1.0 iff GT is empty, else 0.0;
                      otherwise P = tp/|H|, R = tp/|T|,
                      F0.5 = 1.25*P*R / (0.25*P + R)
    * average over entities.

Saves model/decision.json and prints the grid head + breakdowns.
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
from blocking import pack_pairs

def f05_per_entity(tp, nh, nt):
    """Vectorised per-entity macro F_0.5.  Arrays: tp|H∩T|, nh|H|, nt|T|."""
    tp = tp.astype(np.float64)
    nh = nh.astype(np.float64)
    nt = nt.astype(np.float64)
    single = nt == 0
    matched = ~single
    out = np.zeros(len(tp), np.float64)
    out[single] = np.where(nh[single] == 0, 1.0, 0.0)
    m = matched & (nh > 0)
    if m.any():
        p = tp[m] / nh[m]
        r = tp[m] / nt[m]
        den = 0.25 * p + r
        out[m] = np.where(den > 0, (1.25 * p * r) / np.where(den > 0, den, 1.0),
                          0.0)
    # matched but nh == 0 -> out stays 0.0
    return out

def main():
    ap = argparse.ArgumentParser(description="try1 threshold tuning")
    N.add_common_args(ap)
    args = ap.parse_args()
    cache_dir: Path = args.cache_dir
    t0 = time.time()

    vs = np.load(cache_dir / "val_scores.npz")
    gt = np.load(cache_dir / "val_gt.npz")
    val_ids = np.load(cache_dir / "split_val_s1.npy")
    s1, other, score = vs["s1"], vs["other"], vs["score"].astype(np.float64)
    gt_s1, gt_other = gt["s1"], gt["other"]
    print(f"[tune] val entities={len(val_ids):,}  scored candidates={len(s1):,}  "
          f"gt pairs={len(gt_s1):,}", flush=True)
    if len(s1) == 0:
        raise SystemExit("no validation candidates — run blocking first")

    # ---- group candidates by entity (aligned to val_ids rows) -------------
    order = np.lexsort((other, s1))
    s1, other, score = s1[order], other[order], score[order]
    uniq, starts, counts = np.unique(s1, return_index=True, return_counts=True)
    V = len(val_ids)
    M = int(counts.max())
    scores_pad = np.full((V, M), -1.0, np.float64)
    gt_pad = np.zeros((V, M), dtype=bool)
    pos_in_ids = np.searchsorted(val_ids, uniq)
    # per-entity GT membership lookup
    gt_groups: dict = {}
    if len(gt_s1):
        o_g = np.argsort(gt_s1, kind="stable")
        gs, go = gt_s1[o_g], gt_other[o_g]
        g_starts = np.flatnonzero(np.diff(gs)) + 1
        g_starts = np.concatenate(([0], g_starts, [len(gs)]))
        for a, b in zip(g_starts[:-1], g_starts[1:]):
            gt_groups[int(gs[a])] = set(int(x) for x in go[a:b])
    nt = np.zeros(V, np.int64)
    for k in range(len(uniq)):
        j = int(pos_in_ids[k])
        lo, hi = int(starts[k]), int(starts[k] + counts[k])
        scores_pad[j, :hi - lo] = score[lo:hi]
        ent = int(uniq[k])
        gset = gt_groups.get(ent)
        if gset:
            for m_i, o in enumerate(other[lo:hi]):
                if int(o) in gset:
                    gt_pad[j, m_i] = True
        nt[j] = len(gset) if gset else 0
    maxs = scores_pad.max(axis=1)

    # ---- blocking recall on val (ceiling) ---------------------------------
    gt_pack = pack_pairs(gt_s1, gt_other)
    cand_pack = pack_pairs(s1, other)
    hit = np.isin(gt_pack, cand_pack)
    full = {}
    for i, e in enumerate(gt_s1):
        full.setdefault(int(e), True)
        if not hit[i]:
            full[int(e)] = False
    recall_pair = float(hit.mean()) if len(hit) else 1.0
    recall_ent = (float(np.mean(list(full.values()))) if full else 1.0)

    # ---- grid search -------------------------------------------------------
    taus = np.round(np.arange(0.30, 0.951, 0.025), 4)
    gates = [None] + list(np.round(np.arange(0.45, 0.951, 0.05), 4))
    results = []
    for tau in taus:
        mask = scores_pad >= tau
        nh = mask.sum(axis=1)
        tp = (mask & gt_pad).sum(axis=1)
        for gate in gates:
            if gate is None:
                nh2, tp2 = nh, tp
            else:
                active = maxs >= gate
                nh2 = np.where(active, nh, 0)
                tp2 = np.where(active, tp, 0)
            f = f05_per_entity(tp2, nh2, nt)
            results.append((float(f.mean()), float(tau),
                            None if gate is None else float(gate),
                            float((tp2.sum() / nh2.sum())) if nh2.sum() else 1.0,
                            float((tp2.sum() / nt.sum())) if nt.sum() else 1.0))
    results.sort(key=lambda r: (-r[0], -r[3]))  # best F, tie -> higher precision
    best_f, best_tau, best_gate, best_p, best_r = results[0]

    # ---- breakdowns at the chosen decision --------------------------------
    mask = scores_pad >= best_tau
    nh = mask.sum(axis=1)
    tp = (mask & gt_pad).sum(axis=1)
    if best_gate is not None:
        active = maxs >= best_gate
        nh = np.where(active, nh, 0)
        tp = np.where(active, tp, 0)
    f_best = f05_per_entity(tp, nh, nt)

    caches = N.resolve_split_stems("train", args.dataset, cache_dir,
                                   limit=args.limit)
    rows1 = caches[1].row_of(val_ids)
    country = caches[1].get("country", rows1)
    lines = []
    lines.append("=== tune report (validation split) ===")
    lines.append(f"val entities:        {V:,}")
    lines.append(f"val candidates:      {len(s1):,} (mean {len(s1)/V:.1f}/entity)")
    lines.append(f"val GT pairs:        {len(gt_s1):,}   "
                 f"singletons: {int((nt == 0).sum()):,} "
                 f"({100.0*(nt==0).mean():.1f}%)")
    lines.append(f"blocking recall:     pair={recall_pair:.4%}  "
                 f"entity(full)={recall_ent:.4%}   <- recall ceiling")
    lines.append(f"BEST decision:       pair_threshold={best_tau}  "
                 f"entity_gate={best_gate}")
    lines.append(f"BEST macro F_0.5:    {best_f:.5f}   "
                 f"(entity-micro P={best_p:.4f} R={best_r:.4f})")
    lines.append("")
    lines.append("top-10 grid (F, tau, gate, P, R):")
    for r in results[:10]:
        lines.append(f"  F={r[0]:.5f}  tau={r[1]:.3f}  gate={r[2]}  "
                     f"P={r[3]:.4f} R={r[4]:.4f}")
    lines.append("")
    for code, label in N.COUNTRY_LABELS.items():
        sel = country == code
        if sel.any():
            lines.append(f"  F_0.5 [{label}]: {f_best[sel].mean():.5f} "
                         f"(n={int(sel.sum()):,})")
    matched_sel = nt > 0
    if matched_sel.any():
        lines.append(f"  F_0.5 [matched entities]:  {f_best[matched_sel].mean():.5f} "
                     f"(n={int(matched_sel.sum()):,})")
    if (~matched_sel).any():
        lines.append(f"  F_0.5 [singletons]:        {f_best[~matched_sel].mean():.5f} "
                     f"(n={int((~matched_sel).sum()):,})")
    for lo, hi, lab in ((1, 1, "|T|=1"), (2, 2, "|T|=2"),
                        (3, 4, "|T|=3-4"), (5, 99, "|T|=5+")):
        sel = (nt >= lo) & (nt <= hi)
        if sel.any():
            lines.append(f"  F_0.5 [{lab}]: {f_best[sel].mean():.5f} "
                         f"(n={int(sel.sum()):,})")
    report = "\n".join(lines)
    print(report, flush=True)
    (cache_dir / "tune_report.txt").write_text(report)

    decision = {
        "pair_threshold": best_tau,
        "entity_gate": best_gate,
        "val_macro_f05": best_f,
        "val_blocking_pair_recall": recall_pair,
        "val_blocking_entity_recall": recall_ent,
        "grid_top": [
            {"f": r[0], "tau": r[1], "gate": r[2], "p": r[3], "r": r[4]}
            for r in results[:25]
        ],
    }
    model_dir = N.TRY1_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    print(f"[tune] decision saved -> model/decision.json  ({time.time()-t0:.0f}s)",
          flush=True)

if __name__ == "__main__":
    main()
