# Amazon ML Challenge 2026 — Business Entity Resolution

## Problem

We're given business records from **3 independent data sources** (Source 1, Source 2, Source 3)
describing the same real-world businesses, but with **no shared IDs** between sources. Names and
addresses are noisy — abbreviations, legal-suffix variants, typos, transliterations, missing
address components, landmark references, word-order changes, etc.

**Source 1 is the deduplicated reference set.** For every Source 1 entity, we must find all matching
records in Source 2 and/or Source 3. A Source 1 entity may match **zero, one, or many** records.

This is a classic **Entity Resolution (ER)** problem, split into two stages:
1. **Blocking / candidate generation** — cheaply bucket records (e.g. by name+address keys) so we
   only compare plausible pairs instead of the full cross-product. This sets our **recall ceiling** —
   anything not surfaced here can never be matched later.
2. **Matching model** — score each candidate pair and decide true match vs. not, producing the final
   one-row-per-Source-1-entity output.

## Repo structure

.
├── README.md # this file
├── problem_statement.pdf # official problem statement
├── guidelines.pdf # official rules & key instructions
└── student_resource/
├── README.md # short hackathon overview (provided)
├── dataset/
│ ├── train/
│ │ ├── train_source1.tsv
│ │ ├── train_source2.tsv
│ │ ├── train_source3.tsv
│ │ └── train_ground_truth.tsv
│ └── test/
│ ├── test_source1.tsv
│ ├── test_source2.tsv
│ └── test_source3.tsv
└── utils/
├── validate_submission.py # local validator, run before every submission
└── Documentation_template.md # methodology write-up template (final submission)


## Data format

- All files are **tab-separated** (`.tsv`) — addresses and ID lists contain commas, so always read
  with `sep="\t"`.
- Each source file has: `entity_id` (prefixed `S1-`/`S2-`/`S3-`), `business_name`,
  `business_address`, `country`.
- `country` is an **open set** — training covers US and India only; the **test set adds France**,
  unseen in training. Do not hardcode or filter to `{US, India}`.
- `train_ground_truth.tsv`: one row per Source 1 entity → `source1_entity_id`,
  `matched_entity_ids` (comma-separated Source 2/3 IDs, empty if no match / singleton).

## Required outputs

Two files in `output/`:

1. **`matching_results.tsv`** — final matches. **Only file scored on the leaderboard.**
2. **`candidate_pairs.tsv`** — the exact candidate set fed into the matching model at inference
   time (last stage of blocking, not an earlier unfiltered pass). Not scored, but used to audit
   recall ceiling and reduction ratio. Every ID in `matching_results.tsv` must appear here.

Both: one row per Source 1 test entity, empty list for no match, no duplicate IDs, only valid
Source 2/3 test IDs — **every** Source 1 test entity must have exactly one row, France included.

Run `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate
output/candidate_pairs.tsv --test-dir dataset/test` before every submission.

## Evaluation

**Macro-averaged F₀.₅**, computed per Source 1 entity then averaged:

F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)


- Precision-heavy: a **false merge costs ~2× a miss** — when unsure, don't merge.
- **Singletons** (no true match) score 1.0 for predicting empty, 0.0 for predicting anything.
- Public leaderboard = subset of test set (live during challenge); private leaderboard (final
  ranking) = remaining portion, revealed after the challenge ends. Submit predictions for the
  full test set either way.

## Hard constraints

- **No external data lookup** — no entity-resolution APIs, government registries, geocoding
  services, or any outside data augmentation. Provided data only. Violations = disqualification.
- Final model must be **MIT/Apache 2.0 licensed and ≤ 8B parameters**.
- Max 5 leaderboard submissions/day, 3 days total.
- Final submission package (`.zip`): `output/` (both TSVs), `code/business_entity_resolution/`
  (runnable pipeline: `src/`, `README.md`, `requirements.txt`), and a filled-in
  `Documentation_template.md` (methodology, blocking strategy, model architecture/features).

## Approach

*(To be filled in — will be added after we finalize the blocking + matching strategy.)*