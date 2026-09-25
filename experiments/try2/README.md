# try2 — Phase 0: measurement before any key/cap change

Goal: find out where try1's recall went before touching the blocking design.
Measurement-only: keys, caps, and budgets are used exactly as try1 defined
them (the only budget change is the *approved decision rule* that replaces
try1's blind 200M default after first *counting* true sizes).

## Files

| file | role |
|---|---|
| `src/normalize.py` | vendored from try1 (path constants now try2) |
| `src/blocking.py` | vendored from try1 — **read-only in Phase 0** |
| `src/features.py` | vendored from try1 — **read-only in Phase 0** |
| `src/id_check.py` | Step 1: rule out trivial id-ordering solutions (<2 min) |
| `src/count_families.py` | Step 2: true uncapped family sizes + key cache (~6–10 min) |
| `src/build_pool.py` | Step 3: decision rule → annotated candidate pool (~35–60 min) |
| `src/analyze_misses.py` | Step 4: decomposition + recall@K + taxonomy (~10–15 min) |

## How to run (from `experiments/try2/`)

```bash
export DS=../../student_resource/dataset
python src/id_check.py        --dataset $DS                       # step 1
python src/count_families.py  --split train --dataset $DS         # step 2
python src/build_pool.py      --split train --dataset $DS         # step 3
python src/analyze_misses.py  --split train --dataset $DS \
        --try1-cache ../try1/cache                                # step 4
```

All scripts default `--cache-dir` to `experiments/try2/cache` (gitignored).
Every stage writes a timestamped progress log to stdout, checkpoints to
`cache/`, and skips completed work on re-run (`--force` redoes it).

Parallelism: `--workers` (default min(3–4, cpu_count)) for the family/score
stages; peak RAM is kept under ~8 GB by design (one family-side key array per
worker, s1-range bins, chunked joins).

## Decision rule (approved)

`build_pool` reads the counts and prints which path it takes:

- post-cap volume ≤ 600M → **FULL materialization** (budget effectively ∞)
- post-cap volume > 600M → **INFORMED TRIM** to 500M: one uniform scale
  factor across every family/side join budget (equal trim fraction, no
  key-level preferences), exact fractions printed + stored in
  `cache/{split}_pool_meta.json`.

The per-key cap (5000) is unchanged — Phase 0 does not redesign keys.

## What to paste back

1. `id_check.py` output (verdict line matters most)
2. the `count_families.py` table (family sizes)
3. `output/phase0_report.txt`

## Gate

No feature/train/tune changes until blocking entity recall (all GT pairs
found in candidates) reaches ~98% on validation. Phase 1 (transliteration,
new keys, K) is chosen from the taxonomy in that report — not before.
