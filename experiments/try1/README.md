# try1 — blocking union + LightGBM pair classifier

First approach attempt. Entity resolution over the Amazon ML Challenge 2026
dataset: multi-key blocking (8 country-prefixed key families + transliterated
variants) → cheap-score top-K capping → ~40 engineered string/address features
→ LightGBM → threshold + singleton gate tuned directly on macro F₀.₅.

## Layout

```
experiments/try1/
├── src/
│   ├── normalize.py   # text normalization, ID utils, .npy cache builder
│   ├── blocking.py    # stage A key union + stage B top-K cap; recall report
│   ├── features.py    # cheap score, entity aggregates, full feature matrix
│   ├── train.py       # entity split, pair assembly, LightGBM fit, val scores
│   ├── tune.py        # threshold/gate grid on macro F0.5 (validation split)
│   ├── predict.py     # end-to-end test inference + official validator
│   └── inspect.py     # measurement pre-checks (country consistency,
│                      #   canonicalization exact rates, script inventory,
│                      #   cross-script share of true pairs)
├── cache/             # [gitignored] .npy normalization caches, intermediates
├── model/             # [gitignored] lgbm.txt, train_meta.json, decision.json
├── output/            # [gitignored *.tsv] candidate_pairs.tsv, matching_results.tsv
├── requirements.txt
└── README.md          # this file
```

The dataset lives in `student_resource/dataset/{train,test}/` (gitignored —
download from the portal). Every script accepts `--dataset` if it lives
elsewhere; paths otherwise resolve relative to the repo, so you can run from
any working directory.

## Run order (end to end)

```bash
pip install -r experiments/try1/requirements.txt

# 0. measurement pre-checks (~2-4 min) — read the output before proceeding
python3 experiments/try1/src/inspect.py

# 1. blocking on train (~10-30 min; builds caches first run only)
#    -> cache/blocking_report_train.txt  (recall ceiling, reduction ratio)
python3 experiments/try1/src/blocking.py --split train

# 2. train (~30-60 min; features + LightGBM)
#    -> model/lgbm.txt, cache/val_scores.npz, cache/split_val_s1.npy
python3 experiments/try1/src/train.py

# 3. threshold + singleton gate on the validation split (~1-2 min)
#    -> model/decision.json, cache/tune_report.txt
python3 experiments/try1/src/tune.py

# 4. end-to-end test inference (~1-3 h; chunked, bounded RAM)
#    -> output/candidate_pairs.tsv, output/matching_results.tsv
#    runs student_resource/utils/validate_submission.py at the end
python3 experiments/try1/src/predict.py
```

Smoke test on a truncated copy (also useful to sanity-check runtime):
add `--limit 200000` to any command (forces cache rebuild without the flag's
limit — delete `cache/` before a real run afterwards).

## What to paste back after each stage

1. after `inspect.py` — the full output (four sections A-D)
2. after `blocking.py --split train` — `cache/blocking_report_train.txt`
3. after `train.py` — the console log (row counts + early-stop AUC)
4. after `tune.py` — `cache/tune_report.txt`
5. after `predict.py` — console log incl. validator result, plus
   `model/decision.json`

## Expected resource usage (order of magnitude, from EDA row counts)

| stage | wall clock (M-series/8-core) | peak RAM | disk in `cache/` |
|---|---|---|---|
| inspect | 2–5 min | <1 GB | — |
| blocking (train) | 15–40 min (8 family passes) | 3–6 GB | ~8 GB |
| train | 30–60 min | 4–8 GB | ~8 GB (transient) |
| tune | 1–2 min | 1–2 GB | small |
| predict (test) | 1–3 h | 4–8 GB | ~8 GB |

~30–40 GB of free disk is comfortable (caches + outputs + transients).
Lower `--budget` / `--k` on blocking if RAM is tight (recall report shows the
cost).

## Design notes (linking back to the EDA)

- **Postal codes are absent** (92–96% of addresses) → blocking keys are
  name-canopy / locality / house-number based; postal is a feature only.
- **Exact name match covers only ~22%** of true pairs after light
  normalization → 8 key families incl. rare-token canopy (catches
  `dba`/`fka` containment), phonetics, initials (`IAG` cases), and
  transliterated variants for non-Latin Source-2/3 names.
- **Country** is keyed into every blocking key; `inspect.py` (A) verifies the
  100%-consistency assumption on true pairs first (if the check fails, rerun
  blocking after editing `build_all_keys` to drop the `f"{c}|"` prefix).
- **Missing components** get explicit presence/xor feature flags, never a 0
  similarity.
- **France (15% of test S1, unseen in train)**: no country ID is ever a model
  feature — only structural/equality features; French legal suffixes
  (SARL/SAS/EURL/SCI/…) are in the suffix lexicon; de-accenting is built into
  normalization.
- **Singletons (5.6% train)**: `tune.py` grid-searches a pair threshold and an
  entity-level gate against the exact per-entity macro F₀.₅ formula (false
  merges ≈ 2× misses).
- **Split**: entity-level, stratified (country × singleton × cardinality),
  10% validation — no S1 entity leaks across learn/stop/val.

## Constraints compliance

- No external data lookup anywhere: only the provided TSVs + hand-written
  lexicons (legal suffixes, US/India states, street abbreviations).
- Model: LightGBM (MIT, ≪ 8B params). `requirements.txt` carries only
  permissive licenses; the optional transliteration package is commented out
  pending a license check.

## Reproducing outputs

`output/*.tsv` are gitignored (candidate_pairs.tsv alone can exceed 1 GB).
Regenerate with `predict.py` from the same data + seed; everything is
deterministic (fixed seeds, deterministic 64-bit key hashing).
