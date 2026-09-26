# try2 — Phase 0: measurement before any key/cap change

Goal: find out where try1's recall went before touching the blocking design.
Measurement-only: keys, caps, and budgets are used exactly as try1 defined
them (the only budget change is the *approved decision rule* that replaces
try1's blind 200M default after first *counting* true sizes).

## Phase 1 — measurement → evidence-driven fixes (current)

Steps 0–4 are **shipped and measured** (real results: alignment clean, tracer
CAP 99.72% / NOKEY 0.28% / BUG 0, cap curve, rescore delivered@80
76.06→76.65%). Steps 5–6 below are the rest of Phase 1.

```bash
export DS=../../student_resource/dataset
pip install -r requirements.txt     # adds indic-transliteration (MIT), once

# ---- Step 5 cycle (ONE expensive rebuild; ~2-3 h, dominated by stage D) ----
python src/trace_misses.py   --split train --dataset $DS   # ~1-2 min (updated: stamps best-family per victim)
python src/cap_plan.py       --split train --dataset $DS   # ~1-2 min -> {split}_cap_plan.json (per-family caps)
python src/count_families.py --split train --dataset $DS   # ~6-8 min; auto-rebuilds keys on KEY_VERSION bump, applies plan
python src/build_pool.py     --split train --dataset $DS   # FULL run (never --rescore across a key/cap change);
                                                           # stage-D ETA was 3.3 h @3 workers measured —
                                                           # defaults now A=4 / D=6 (~1.7 h stage D)

# ---- Step 6: gate + fresh measurement on the NEW pool ----
python src/trace_misses.py   --split train --dataset $DS   # fresh classification (~1-2 min)
python src/gate_check.py     --split train --dataset $DS   # ~60-90 s -> PASS/FAIL at 98% entity in-pool
python src/analyze_misses.py --split train --dataset $DS --try1-cache ../try1/cache
python src/cap_plan.py       --split train --dataset $DS   # residue report + seed for the next cycle
```

Paste back: **gate output**, both `output/cap_plan.txt`, both tracer
summaries, and the analyze decomposition + recall@K.

Mechanics worth knowing:
- `KEY_VERSION` in `blocking.py` — bump on any emission change;
  `count_families` auto-rebuilds keys when counts are stale, `build_pool`
  refuses to run stage A against stale counts (`plan_sig` also invalidates
  stage A whenever the cap plan changes).
- `--rescore` only for score/rank changes on an UNCHANGED pool; it refuses
  if keys or caps moved since stage A.
- **Gate:** stage-A entity recall ≥98% (all GT pairs in pool, any rank)
  before any feature/train/tune work. Cycles repeat (keys/caps) until it
  passes, then Phase 2.

## Phase 0 — measurement (complete)

| file | role |
|---|---|
| `src/normalize.py` | vendored from try1 (path constants now try2) |
| `src/blocking.py` | vendored from try1 — read-only in Phase 0 |
| `src/features.py` | vendored from try1 + `compute_cheap_scores_v2` (Phase 1) |
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

## Gate (standing)

No feature/train/tune changes until stage-A entity recall (all GT pairs
present in candidates) reaches ~98% on validation. Key changes (Step 5) are
selected from the Phase-1 tracer output — never before it.
