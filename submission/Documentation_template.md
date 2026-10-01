# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name — fill in]  
**Team Members:** [List all team members — fill in]  
**Submission Date:** 2026-10-01 (package assembled; result submitted during the active window)

---

## 1. Executive Summary
*Approach: multi-key blocking union → cheap-score top-K capping → ~40 engineered
string/address features → LightGBM → threshold + singleton gate tuned directly
on macro F₀.₅. Eight country-prefixed hashed key families (name-sorted,
rare-token canopy, phonetic, initials, house-number, address-token, and
address+name combinations, plus transliterated variants for non-Latin names)
reduce 10¹² cross-products to a manageable candidate set; a gradient-boosted
tree classifier then decides each pair. Key innovations: (a) deterministic
per-bucket cartesian-cap blocking semantics that keep the smallest, most
selective buckets under a fixed volume budget, and (b) entity-level aggregate
features + an entity singleton gate tuned end-to-end against the exact
per-entity macro F₀.₅ scoring formula. **Public leaderboard: 0.813.**

---

## 2. Methodology

### 2.1 Problem Analysis
*EDA-driven insights used by the solution:*
- **Postal codes are absent** from 92–96% of addresses → PIN cannot be a
  blocking key or a similarity feature; blocking must be name- and
  locality-based, postal a presence flag only.
- **Exact name match is insufficient**: normalized name-sorted equality
  covers 50.00% of true pairs (name-core 47.68%), full address-normalized
  equality only 12.13% — the other half is token subsets, suffix/order
  variants, noise, and script differences.
- **Strong address bridges exist**: house number equality 78.09%, state
  equality 95.30% — usable as keys and features even without postal.
- **Non-Latin counterparts** are 7.217% of true pairs (Source-2/3 names in
  Devanagari and other scripts) → cross-script name keys via optional
  transliteration, with graceful degradation if the package is absent.
- **France appears only in test** (≈15% of test S1 entities, unseen in
  train) → country identity is never a model feature; French legal suffixes
  (SARL/SAS/EURL/SCI/…) are in the suffix lexicon and de-accenting is built
  into normalization.
- **Singletons ≈5.6% of train S1 entities** → the decision rule needs an
  explicit entity-level gate, not just a pair threshold.
- Missing fields (address, house number, name components) get explicit
  presence/xor features, never a zero similarity.

### 2.2 Solution Strategy
Classic **blocking + classifier** pipeline, fully deterministic end to end
(fixed seed 42, 64-bit key hashing): normalize → multi-family key union with
per-bucket cap and total-volume budget → top-80 candidates per S1 entity by a
cheap name score → engineered features with entity aggregates → LightGBM →
τ/singleton-gate grid search on validation macro F₀.₅ → test inference.

**Approach Type:** Blocking + Classifier  
**Core Innovation:** Country-prefixed multi-family key union with
per-bucket cartesian-cap + smallest-buckets-first volume budget (selective
buckets are never traded away for large ones), combined with entity-level
aggregate features and a singleton gate tuned against the exact challenge
metric.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used** (all country-prefixed, 64-bit hashed, emitted per
  row and joined across Source-1 × {Source-2, Source-3}):
  - `sig` — token-sorted, suffix-stripped normalized name
  - `pfx` — first 3 sorted tokens + last token (order/partial tolerant)
  - `tok` — every rare-ish core name token (containment: dba/fka/alias)
  - `phon` — phonetic key of first+last core token (typos/transliteration)
  - `ini` — initials of ≥3 core tokens ("IAG" cases)
  - `hn` — house number; `combo` — house number | first name token
  - `atok` — address tokens, len ≥ 4 (locality/street bridge)
  - transliterated variants of sig/pfx/tok/phon/ini for non-Latin rows
    (optional `indic-transliteration` package, license-gated; degrades to
    raw-script + address/token bridging when absent)
- **Candidate pairs generated:** the exact set ships as
  `output/candidate_pairs.tsv` in this package (stage A union, then the top
  **80** candidates per S1 entity ranked by
  `max(token_jaccard, token_containment)` on token-sorted core names).
- **How true matches were kept as far as possible:**
  - redundant union across 8 independent families — a pair survives if *any*
    family keys it;
  - every run prints a recall report against ground truth (our run: pair
    recall **77.3010%**, entity all-found **51.4488%**), so the ceiling is
    measured, not assumed;
  - cap/budget semantics keep the **smallest** buckets first (most selective
    keys survive truncation);
  - country is verified 100%-consistent on true pairs before keying.

---

## 4. Matching Model

**Features used (~40, all engineered in `src/features.py`):**
- Name features: token Jaccard / containment, normalized similarity
  (Indel-based), phonetic match, initials overlap, token-level best
  alignments, suffix-type agreement, script agreement.
- Address features: address-token overlap, house-number equality, state
  equality, postal presence flag, address presence/xor flags.
- Other: entity-level aggregates (max/mean of pair scores within an entity's
  candidate star), candidate-rank features from the cheap score, missing-
  component presence flags (never zeros).

**Model type:** LightGBM (`LGBMClassifier`, lightgbm==4.4.0, MIT) — binary
pair classifier with entity-level 10% validation split, stratified on
country × singleton × cardinality; early stopping (our run: 6,252 s,
best_iter 2984).

**Threshold selection method:** grid search over pair threshold τ and an
entity singleton gate on the validation split, scored with the exact
per-entity macro F₀.₅ formula (precision-weighted, β = 0.5; false merges
≈ 2× missed matches). Best: **τ = 0.925** + gate → validation macro
F₀.₅ = **0.85303**.

---

## 5. Results & Error Analysis

- **F₀.₅ Score (macro):** validation **0.85303** (micro P 0.97176,
  R 0.72102); **public leaderboard 0.813** (this submission).
- **Known limitation — blocking budget truncation:** stage A drops whole
  key buckets whose cartesian product exceeds `--per-key-cap 5000` and
  truncates total stage-A volume at `--budget 200,000,000` pairs,
  keeping the smallest buckets first. True pairs that *only* co-occur in
  oversized buckets were never candidates; this bounds end-to-end recall
  (measured blocking pair recall 77.30%, entity all-found 51.45%) and is
  the largest known gap between validation (0.853) and leaderboard (0.813).
- **Common false positives (wrong merges):** near-identical company names
  across genuinely different entities (branches/chains sharing address
  components); mitigated by address/model features and the singleton gate.
- **Common false negatives (missed matches):** pairs whose only shared key
  lived in a truncated bucket (large-bucket localities, very common names),
  token-disjoint name variants, and cross-script pairs when transliteration
  is disabled.

---

## 6. Conclusion
A single deterministic LightGBM pipeline — 8-family hashed blocking with
budgeted cap semantics, ~40 engineered features, and F₀.₅-native threshold +
singleton-gate tuning — reached **0.813 on the public leaderboard** with no
external data and a fully permissive-license stack. The main lesson: candidate
recall under cap/budget truncation is the binding constraint, so the next
iteration targets tighter composite keys (state/house-number qualified, name
bigrams) rather than looser caps; error analysis also shows identical-name
pairs need stronger address/model discrimination.

---

## Appendix

### A. Code Artefacts
Ships in this zip under `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── src/
│   ├── normalize.py    # normalization, IDs, cache builder
│   ├── inspect.py      # EDA pre-checks (run first)
│   ├── blocking.py     # stage A keys + stage B top-80; recall report
│   ├── features.py     # cheap score, aggregates, feature matrix
│   ├── train.py        # split, LightGBM fit, validation scores
│   ├── tune.py         # τ + singleton-gate grid on macro F₀.₅
│   └── predict.py      # test inference → output/*.tsv + validator
├── README.md           # exact end-to-end reproduction commands & flags
└── requirements.txt    # pinned dependencies
```

Entry points to reproduce `output/matching_results.tsv` and
`output/candidate_pairs.tsv`: `inspect.py` → `blocking.py --split train` →
`train.py` → `tune.py` → `predict.py` (full commands, real flag values, and
output locations are in the package README). The two submitted TSVs are in
the top-level `output/` folder of this zip.

### B. Additional Results
| stage | measured result |
|---|---|
| blocking (train) | pair recall 77.3010%, entity all-found 51.4488%, top-80 per S1 |
| train | 6,252 s, best_iter 2984, seed 42 |
| tune | macro F₀.₅ 0.85303 @ τ = 0.925 (micro P 0.97176 / R 0.72102) |
| predict | 13,289 s; 192,688 S1 entities with zero candidates (11.1%) |
| leaderboard | **0.813** |

---

**Note:** Sections were filled in for the try1 approach while keeping the
template's structure; team name/members are left to be completed by the team.
