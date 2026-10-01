# Business Entity Resolution — submission code (try1)

Reproduction instructions for the Amazon ML Challenge 2026 Business Entity
Resolution submission: multi-key blocking union + cheap-score top-K capping +
~40 engineered string/address features + LightGBM, with threshold and
singleton gate tuned directly on macro F_0.5. The submitted
`output/matching_results.tsv` came from this exact pipeline (public
leaderboard **0.813**).

This file documents the commands and flag values that were actually used.
It is documentation — reproducing requires the challenge dataset (not
included in the submission, per the rules).

## Layout

```
code/business_entity_resolution/
├── src/
│   ├── normalize.py   # text normalization, ID utils, .npy cache builder
│   ├── inspect.py     # measurement pre-checks (country consistency,
│   │                  #   canonicalization exact rates, script inventory)
│   ├── blocking.py    # stage A key union + stage B top-K cap; recall report
│   ├── features.py    # cheap score, entity aggregates, full feature matrix
│   ├── train.py       # entity split, pair assembly, LightGBM fit, val scores
│   ├── tune.py        # threshold/singleton-gate grid on macro F0.5
│   └── predict.py     # end-to-end test inference + official validator
├── README.md          # this file
└── requirements.txt   # pinned dependencies
```

Entry point chain: `inspect.py` → `blocking.py` → `train.py` → `tune.py` →
`predict.py`. Every script accepts `--dataset`, `--cache-dir`, `--limit`
(defaults resolve relative to this package).

## 1. Environment setup

- Python **3.9 – 3.12** (the `numpy==1.26.4` pin has no wheels on 3.13).
- RAM: ~8 GB peak; disk: ~30–40 GB free (normalization caches + outputs).

```bash
pip install -r requirements.txt
```

## 2. Dataset

Place the official challenge data at `student_resource/dataset/{train,test}/`
(repo layout), or anywhere and pass `--dataset /path/to/dataset` to every
command below. No other data is used — lexicons (legal suffixes, US/India
states, street abbreviations) are hand-written and ship inside `src/`.

## 3. Exact commands, in order (the flags below are the values actually used)

```bash
# 0. measurement pre-checks (~2–5 min) — read the four sections A–D first
python3 src/inspect.py

# 1. blocking on train (~15–40 min; builds normalization caches on first run)
#    flags actually used (these are also the defaults):
#        --split train  -k 80  --per-key-cap 5000  --budget 200000000
python3 src/blocking.py --split train
#    -> cache/blocking_report_train.txt   (recall ceiling + reduction ratio)
#    -> cache/candidates_train.npz        (stage-B candidate set)

# 2. train (~30–60 min; our run: 6,252 s, best_iter 2984)
#    flags actually used (also the defaults):
#        --neg-cap 6  --val-frac 0.10  --stop-frac 0.15  --seed 42
python3 src/train.py
#    -> model/lgbm.txt, model/train_meta.json,
#       cache/val_scores.npz, cache/split_val_s1.npy

# 3. threshold + singleton gate on the validation split (~1–2 min)
#    grid-searches tau + entity gate against macro F0.5; our best: tau = 0.925
python3 src/tune.py
#    -> model/decision.json, cache/tune_report.txt

# 4. end-to-end test inference (~1–4 h; our run: 13,289 s)
#    flags actually used (also the defaults):
#        --chunk 1000000  --per-key-cap 5000  --budget 200000000
python3 src/predict.py
#    -> output/candidate_pairs.tsv
#    -> output/matching_results.tsv
#    -> runs student_resource/utils/validate_submission.py at the end
```

If you run from the original repository checkout instead of this extracted
package, prefix the scripts with `experiments/try1/` (e.g.
`python3 experiments/try1/src/blocking.py --split train`) — that is the
layout the commands above were originally executed in, with identical flags.

## 4. Where outputs land

| artifact | stage | location |
|---|---|---|
| normalization `.npy` caches, `blocking_report_train.txt`, `candidates_train.npz`, `val_scores.npz` | 0–3 | `cache/` (next to `src/`, i.e. `code/business_entity_resolution/cache/`) |
| `lgbm.txt`, `train_meta.json`, `decision.json` | 2–3 | `model/` |
| **`candidate_pairs.tsv`, `matching_results.tsv`** | 4 | `output/` — in the original run: `experiments/try1/output/`; from this package: `code/business_entity_resolution/output/` |

Before zipping for upload, place the two TSVs in the package's top-level
`output/` directory (the structure the validator/leaderboard expects).

### Validator note

`predict.py` finishes by invoking the official
`student_resource/utils/validate_submission.py` (resolved two levels above
this package). In a full repository checkout it runs automatically and
prints `validator PASS`. In the extracted submission package alone that path
does not exist — outputs are already written by then, so either add
`--skip-validator` or restore the `student_resource/utils/` folder next to
the package root.

## 5. What this configuration produced (actual run)

- blocking (train): pair recall **77.3010%**, entity all-found **51.4488%**
  (see `cache/blocking_report_train.txt` after step 1)
- train: 6,252 s, `best_iter 2984`
- tune: best validation macro **F_0.5 = 0.85303** at τ = 0.925
  (micro P 0.97176, R 0.72102)
- predict: 13,289 s; 192,688 S1 entities with zero candidates (11.1%)
- public leaderboard: **0.813**

## 6. Determinism & compliance

- Fixed seeds (42) and deterministic 64-bit key hashing: rerunning step 4
  from the same data + `model/` regenerates the TSVs bit-for-bit.
- No external data lookup anywhere; only the provided TSVs + hand-written
  lexicons. Model: LightGBM (MIT, far below the 8B-param cap).
- `requirements.txt` carries only permissive licenses; the optional
  transliteration package stays commented out pending team license sign-off
  (see the notes there).
