#!/usr/bin/env python3
"""
Step-1 EDA for the Amazon ML Challenge 2026 — Business Entity Resolution.

Run it locally against the real dataset (downloaded from the official portal into
student_resource/dataset/, which is gitignored). It answers:

  1. Row counts + country distribution per file (train/test x source1/2/3)
  2. Ground-truth analysis: singleton rate, matches/S1 stats, S2-only vs S3-only
     vs both mix, S1 entities with no GT row, S2/S3 IDs claimed by >1 S1
     ("S1 is deduplicated" sanity check), GT IDs missing from the source files
  3. Exact-match rates on known true pairs (raw name / normalized name /
     raw address / normalized address) — sizes the "easy" fraction
  4. Missing-field rates (empty name/address/country, no postal-like code)
  5. Encoding report (UTF-8 validity, non-ASCII/accented lines — France check)
  6. ~20 random true match pairs and ~20 near-miss non-pairs (same postal code
     or same first name token, but NOT in the ground truth), printed for
     eyeballing the actual noise patterns
  7. Train-vs-test structural drift + the France subset (counts + sample rows)
  8. Rough per-file size / memory projections (chunked vs in-memory guidance)

Usage (paths resolve relative to THIS file, so it works from anywhere):

    python3 student_resource/scripts/eda.py                 # full run
    python3 student_resource/scripts/eda.py --limit 200000  # quick smoke test

Design:
- Standard library only (no pandas/numpy) — no install step, runs on any
  Python 3.8+.
- Every source file is streamed line-by-line; we never hold a whole TSV in
  memory. Peak RAM is dominated by small dictionaries/sets (tens of MB) —
  safe on a low-memory machine.
- All randomness is seeded (--seed, default 42) => reproducible output.
- Only SMALL samples are printed (<= ~20 rows per section).
"""

import argparse
import os
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Paths: student_resource/scripts/eda.py  ->  student_resource/dataset/...
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

SOURCE_FILES = [
    ("train", TRAIN_DIR / "train_source1.tsv"),
    ("train", TRAIN_DIR / "train_source2.tsv"),
    ("train", TRAIN_DIR / "train_source3.tsv"),
    ("test", TEST_DIR / "test_source1.tsv"),
    ("test", TEST_DIR / "test_source2.tsv"),
    ("test", TEST_DIR / "test_source3.tsv"),
]
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

EXPECTED_HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry"

# A "postal-like" token: 5-digit (US ZIP / French code postal, optional +4),
# or 6-digit (India PIN). Purely heuristic — house numbers of 5-6 digits can
# false-positive; that is fine for an EDA presence rate.
POSTAL_RE = re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)|(?<!\d)\d{6}(?!\d)")

# Normalizer used for the "normalized exact-match" rates. Keeps unicode
# letters (accents preserved); folds case, punctuation, '&'->'and', whitespace.
NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def norm_text(s: str) -> str:
    """NFKC + casefold + '&'->'and' + punctuation->space + whitespace collapse."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.replace("&", " and ")
    s = NON_WORD_RE.sub(" ", s)
    return " ".join(s.split())


def first_token(norm_name: str) -> str:
    return norm_name.split()[0] if norm_name else ""


def postal_like(address: str) -> str:
    """First postal-looking token in an address, or '' if none."""
    m = POSTAL_RE.search(address)
    return m.group(0) if m else ""


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def trunc(s: str, n: int = 90) -> str:
    s = s.replace("\t", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def hdr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74, flush=True)


def looks_french(country: str) -> bool:
    return country.strip().upper().startswith("FR")  # 'France', 'FR', 'FRA', ...


# --------------------------------------------------------------------------
# Ground truth loader (small file, streamed).
# --------------------------------------------------------------------------
def load_ground_truth(path: Path):
    """Return (gt, report) where gt: {s1_id: [matched_ids]} (empty list = singleton)."""
    gt = {}
    issues = Counter()
    header = None
    with open(path, "rb") as f:
        first = True
        for raw in f:
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                issues["utf8_decode_errors"] += 1
                line = raw.decode("utf-8", "replace")
            line = line.rstrip("\n").rstrip("\r")
            if first:
                first = False
                header = line
                continue
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                issues["rows_without_exactly_2_columns"] += 1
                if len(parts) < 2:
                    parts = parts + [""] * (2 - len(parts))
            s1 = parts[0].strip()
            ids = [x.strip() for x in parts[1].split(",") if x.strip()]
            if not s1:
                issues["rows_with_empty_source1_id"] += 1
                continue
            if s1 in gt:
                issues["duplicate_source1_rows"] += 1
                gt[s1].extend(ids)
            else:
                gt[s1] = ids
            if len(ids) != len(set(ids)):
                issues["duplicate_ids_within_a_row"] += 1
            for m in ids:
                if m.startswith("S1-"):
                    issues["S1_prefixed_id_in_matches"] += 1
                elif not m.startswith(("S2-", "S3-")):
                    issues["unrecognized_id_prefix"] += 1
    report = {"header": header, "issues": issues}
    return gt, report


# --------------------------------------------------------------------------
# Streaming reader for the 7 source files (handles encoding fallback safely).
# Fills a caller-owned stats dict; yields (entity_id, name, address, country).
# --------------------------------------------------------------------------
def new_file_stats(path: Path) -> dict:
    return {
        "path": path,
        "rows": 0,
        "header": None,
        "header_ok": False,
        "utf8_errors": 0,
        "first_utf8_error_line": None,
        "nonascii_rows": 0,
        "nonascii_examples": [],   # up to 3 (line_no, text)
        "bad_tab_rows": 0,         # rows where field count != 4
        "countries": Counter(),
        "missing_name": 0,
        "missing_addr": 0,
        "missing_country": 0,
        "addr_without_postal": 0,  # heuristic 5-6 digit token missing
        "addr_nonempty": 0,
        "first_rows": [],          # up to 2 sample data rows (truncated)
        "elapsed_s": 0.0,
        "limited": False,
    }


def stream_rows(path: Path, st: dict, limit=None):
    """Yield (entity_id, name, address, country) for each data row; fills st."""
    with open(path, "rb") as f:
        first_line = True
        n_data = 0
        for raw in f:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                st["utf8_errors"] += 1
                if st["utf8_errors"] == 1:
                    st["first_utf8_error_line"] = st["rows"] + n_data  # approx line no
                text = raw.decode("utf-8", "replace")
            line = text.rstrip("\n").rstrip("\r")
            if first_line:
                first_line = False
                st["header"] = line
                st["header_ok"] = line == EXPECTED_HEADER
                continue
            if limit is not None and n_data >= limit:
                break
            if not line.isascii():
                st["nonascii_rows"] += 1
                if len(st["nonascii_examples"]) < 3:
                    st["nonascii_examples"].append((n_data + 1, trunc(line, 200)))
            if line.count("\t") != 3:
                st["bad_tab_rows"] += 1
            fields = line.split("\t", 3)
            while len(fields) < 4:
                fields.append("")
            n_data += 1
            st["rows"] = n_data
            yield fields[0], fields[1], fields[2], fields[3]
    st["limited"] = limit is not None
    st["rows"] = n_data


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Step-1 EDA for the entity-resolution challenge.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max data rows to read PER SOURCE FILE (smoke test). Default: all rows.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: %(default)s).")
    ap.add_argument("--exact-sample", type=int, default=50000,
                    help="True pairs to sample for exact-match rates (default: %(default)s).")
    ap.add_argument("--seed-entities", type=int, default=60,
                    help="S1 entities used as seeds for near-miss mining (default: %(default)s).")
    ap.add_argument("--pairs-print", type=int, default=20, help="True pairs to print (default: 20).")
    ap.add_argument("--miss-print", type=int, default=20, help="Near-miss pairs to print (default: 20).")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- 0. sanity: dataset present? -------------------------------------
    hdr("0) SETUP — dataset location and file sizes")
    print(f"dataset dir: {DATASET_DIR}")
    expected = [GT_PATH] + [p for _, p in SOURCE_FILES]
    missing = [p for p in expected if not p.is_file()]
    if not DATASET_DIR.is_dir() or missing:
        print("\nDATASET NOT FOUND. Expected these files (relative to the repo):")
        for p in expected:
            print(f"  {'OK  ' if p.is_file() else 'MISS'} {p.relative_to(DATASET_DIR.parents[1])}")
        print(
            "\nPlace the official portal download under student_resource/dataset/\n"
            "(that directory is intentionally gitignored). Then re-run:\n"
            "    python3 student_resource/scripts/eda.py --limit 200000   # smoke test first\n"
            "    python3 student_resource/scripts/eda.py                   # full run"
        )
        return 2
    for split, p in SOURCE_FILES:
        mb = os.path.getsize(p) / 1e6
        print(f"  {split:5s}  {p.name:22s}  {mb:8.1f} MB")
    gt_mb = os.path.getsize(GT_PATH) / 1e6
    print(f"  train  {GT_PATH.name:22s}  {gt_mb:8.1f} MB")
    if args.limit:
        print(f"\n*** --limit={args.limit}: counts/rates below are PARTIAL (first "
              f"{args.limit} data rows of each source file) ***")

    # ---- 1. ground truth --------------------------------------------------
    hdr("1) GROUND TRUTH — train_ground_truth.tsv")
    gt, gt_report = load_ground_truth(GT_PATH)
    print(f"header: {gt_report['header']!r}")
    gt_issues = gt_report["issues"]
    if gt_issues:
        print(f"format issues: {dict(gt_issues)}")
    else:
        print("format issues: none detected")

    n_entities = len(gt)
    singleton_rows = sum(1 for ids in gt.values() if not ids)
    matched_rows = n_entities - singleton_rows
    counts = [len(ids) for ids in gt.values()]
    counts_matched = [c for c in counts if c > 0]
    print(f"S1 entities listed in GT: {n_entities}")
    print(f"  explicit singletons (empty match list): {singleton_rows} "
          f"({pct(singleton_rows, n_entities)})")
    print(f"  with >=1 match:                         {matched_rows} "
          f"({pct(matched_rows, n_entities)})")
    if counts:
        print(f"  matches/S1 — all rows:   mean={statistics.mean(counts):.3f}  "
              f"median={statistics.median(counts)}  max={max(counts)}")
    if counts_matched:
        print(f"  matches/S1 — matched only: mean={statistics.mean(counts_matched):.3f}  "
              f"median={statistics.median(counts_matched)}  max={max(counts_matched)}")
    buckets = Counter()
    for c in counts:
        if c == 0:
            buckets["0"] += 1
        elif c == 1:
            buckets["1"] += 1
        elif c == 2:
            buckets["2"] += 1
        elif c <= 4:
            buckets["3-4"] += 1
        elif c <= 10:
            buckets["5-10"] += 1
        else:
            buckets["11+"] += 1
    print("  histogram:", {k: buckets[k] for k in ["0", "1", "2", "3-4", "5-10", "11+"] if k in buckets})

    mix = Counter()
    for ids in gt.values():
        has2 = any(m.startswith("S2-") for m in ids)
        has3 = any(m.startswith("S3-") for m in ids)
        if not ids:
            mix["empty (singleton)"] += 1
        elif has2 and has3:
            mix["both S2 and S3"] += 1
        elif has2:
            mix["S2 only"] += 1
        elif has3:
            mix["S3 only"] += 1
        else:
            mix["unrecognized prefixes"] += 1
    print("  match-source mix:", dict(mix))

    # Cross-links: same matched ID claimed by more than one S1.
    id_owner, cross_links = {}, {}
    for s1, ids in gt.items():
        for m in dict.fromkeys(ids):  # de-dup within row for this analysis
            if m in id_owner:
                if m not in cross_links:
                    cross_links[m] = [id_owner[m], s1]
                else:
                    cross_links[m].append(s1)
            else:
                id_owner[m] = s1
    print(f"matched IDs claimed by >1 S1 entity: {len(cross_links)}")
    for m, owners in list(cross_links.items())[:10]:
        print(f"  {m} <- {owners}")
    if not cross_links:
        print("  (none — consistent with 'S1 is deduplicated / pure 1-to-many', "
              "to be confirmed against the source files below)")

    # Sampled pairs + seeds (needed BEFORE streaming the source files).
    pairs = [(s1, m) for s1, ids in gt.items() for m in ids]
    exact_sample = rng.sample(pairs, min(args.exact_sample, len(pairs)))
    print_pairs = exact_sample[: args.pairs_print]
    eligible_seeds = [s1 for s1, ids in gt.items() if ids]
    seed_ids = set(rng.sample(eligible_seeds, min(args.seed_entities, len(eligible_seeds))))
    print(f"sampled {len(exact_sample)} true pairs for exact-match rates (seed={args.seed})")
    print(f"mined near-misses against {len(seed_ids)} seed S1 entities")

    capture_ids = set(seed_ids)
    for s1, m in exact_sample:
        capture_ids.add(s1)
        capture_ids.add(m)
    captured = {}  # id -> (name, address, country); filled while streaming train files

    # Pre-index every GT-referenced matched ID so we can verify they exist.
    matched_ids_seen = {m: False for m in id_owner}

    # ---- 2. stream every source file once ---------------------------------
    hdr("2) STREAMING PASS (progress lines) — one pass per file")
    t_start = time.time()
    file_stats = {}   # filename -> stats dict
    train1_ids = set()
    seed_postal = defaultdict(set)   # postal  -> {seed S1 ids}
    seed_token = defaultdict(set)    # name-1st-token -> {seed S1 ids}
    near_miss = []                   # (reason, s1, other_row, s1_row)
    near_miss_seen = set()
    NEAR_MISS_CAP = 400
    france_samples = []              # (file, id, name, address)

    for split, path in SOURCE_FILES:
        st = new_file_stats(path)
        print(f"[streaming] {path.name} ...", flush=True)
        t0 = time.time()
        is_train1 = path.name == "train_source1.tsv"
        is_train23 = path.name in ("train_source2.tsv", "train_source3.tsv")
        is_test = split == "test"
        for eid, name, addr, country in stream_rows(path, st, args.limit):
            # --- common per-row bookkeeping ---
            c = country.strip()
            st["countries"][c] += 1
            if not name.strip():
                st["missing_name"] += 1
            if not addr.strip():
                st["missing_addr"] += 1
            if not c:
                st["missing_country"] += 1
            if addr.strip():
                st["addr_nonempty"] += 1
                if not postal_like(addr):
                    st["addr_without_postal"] += 1
            if len(st["first_rows"]) < 2:
                st["first_rows"].append((eid, trunc(name, 60), trunc(addr, 60), c))

            # --- file-specific ---
            if is_train1:
                train1_ids.add(eid)
                if eid in capture_ids:
                    captured[eid] = (name, addr, c)
                if eid in seed_ids:
                    p = postal_like(addr)
                    if p:
                        seed_postal[p].add(eid)
                    t = first_token(norm_text(name))
                    if t:
                        seed_token[t].add(eid)
            elif is_train23:
                if eid in matched_ids_seen:
                    matched_ids_seen[eid] = True
                if eid in capture_ids:
                    captured[eid] = (name, addr, c)
                # near-miss: shares a postal code OR first name token with a seed
                # S1, but is NOT in that S1's ground-truth match list.
                if len(near_miss) < NEAR_MISS_CAP:
                    reasons, cand = [], set()
                    p = postal_like(addr)
                    if p and p in seed_postal:
                        cand |= seed_postal[p]
                        reasons.append("same postal code")
                    t = first_token(norm_text(name))
                    if t and t in seed_token:
                        cand |= seed_token[t]
                        reasons.append("same first name token")
                    for s1 in cand:
                        if eid in gt.get(s1, ()):
                            continue  # a true pair, skip
                        key = (s1, eid)
                        if key in near_miss_seen:
                            continue
                        near_miss_seen.add(key)
                        reason = " + ".join(reasons)
                        s1_row = captured.get(s1)
                        near_miss.append((reason, s1, (eid, name, addr, c), s1_row))
                        if len(near_miss) >= NEAR_MISS_CAP:
                            break
            elif is_test and looks_french(c) and len(france_samples) < 12:
                france_samples.append((path.name, eid, trunc(name, 70), trunc(addr, 70)))
        st["elapsed_s"] = time.time() - t0
        file_stats[path.name] = st
        print(f"  done: {st['rows']:,} rows in {st['elapsed_s']:.1f}s "
              f"({st['utf8_errors']} utf8 errors, {st['nonascii_rows']} non-ASCII rows)", flush=True)
    total_s = time.time() - t_start
    print(f"\nall 6 source files streamed in {total_s:.1f}s total", flush=True)

    # ---- 3. inventory + memory projection ---------------------------------
    hdr("3) FILE INVENTORY, ROW COUNTS, MEMORY PROJECTION")
    print(f"{'file':24s} {'MB':>8s} {'rows':>10s} {'B/row':>7s} "
          f"{'pandas-object est.':>20s}  guidance")
    for split, path in SOURCE_FILES:
        st = file_stats[path.name]
        size_b = os.path.getsize(path)
        avg = size_b / max(st["rows"], 1)
        # Rough heuristic: one object-dtype pandas row costs ~= 2.5x its raw
        # text size (python object overhead). ASSUMPTION, order-of-magnitude only.
        est_gb = st["rows"] * avg * 2.5 / 1e9
        guide = ("fits in RAM (as one DF)" if est_gb < 2.0
                 else "USE CHUNKED/STREAMING")
        ok = "ok" if st["header_ok"] else f"HEADER MISMATCH: {st['header']!r}"
        print(f"{path.name:24s} {size_b/1e6:8.1f} {st['rows']:10,d} {avg:7.0f} "
              f"{est_gb:19.2f}G  {guide}")
        print(f"{'':24s} header: {ok}; rows with unexpected field count: {st['bad_tab_rows']}")

    # ---- 4. country distribution ------------------------------------------
    hdr("4) COUNTRY DISTRIBUTION PER FILE")
    for split, path in SOURCE_FILES:
        st = file_stats[path.name]
        top = st["countries"].most_common(12)
        rendered = ", ".join(f"{c or '(empty)'}={n:,}" for c, n in top)
        extra = len(st["countries"]) - len(top)
        if extra > 0:
            rendered += f", ... (+{extra} more labels)"
        print(f"{path.name:24s} {rendered}")
    train_countries = set()
    test_countries = set()
    for split, path in SOURCE_FILES:
        (train_countries if split == "train" else test_countries).update(
            file_stats[path.name]["countries"])
    print(f"\ntrain country labels: {sorted(train_countries)}")
    print(f"test  country labels: {sorted(test_countries)}")
    only_test = test_countries - train_countries
    if only_test:
        print(f"labels seen ONLY in test (unseen in training): {sorted(only_test)}")

    # ---- 5. missing-field rates -------------------------------------------
    hdr("5) MISSING-FIELD RATES PER FILE")
    print(f"{'file':24s} {'empty name':>11s} {'empty addr':>11s} {'empty ctry':>11s} "
          f"{'no-postal*':>11s}")
    for split, path in SOURCE_FILES:
        st = file_stats[path.name]
        r = st["rows"]
        print(f"{path.name:24s} {pct(st['missing_name'], r):>11s} "
              f"{pct(st['missing_addr'], r):>11s} {pct(st['missing_country'], r):>11s} "
              f"{pct(st['addr_without_postal'], st['addr_nonempty']):>11s}")
    print("* no-postal = share of NON-EMPTY addresses without a 5-6 digit postal-like token")
    print("  (heuristic; denominator = rows with a non-empty address)")

    # ---- 6. encoding -------------------------------------------------------
    hdr("6) ENCODING CHECK (UTF-8 validity, accented/non-ASCII lines)")
    for split, path in SOURCE_FILES:
        st = file_stats[path.name]
        status = "valid UTF-8" if st["utf8_errors"] == 0 else (
            f"NOT valid UTF-8 — {st['utf8_errors']} line(s) failed strict decode "
            f"(first near line {st['first_utf8_error_line']}); decoded with U+FFFD replacement")
        print(f"{path.name:24s} {status}")
        print(f"{'':24s} non-ASCII lines: {st['nonascii_rows']:,} "
              f"({pct(st['nonascii_rows'], st['rows'])} of rows)")
        for line_no, sample in st["nonascii_examples"]:
            print(f"{'':26s} e.g. line {line_no}: {sample}")
    with open(GT_PATH, "rb") as f:
        gt_bytes = f.read()
    try:
        gt_bytes.decode("utf-8")
        print(f"{GT_PATH.name:24s} valid UTF-8")
    except UnicodeDecodeError as e:
        print(f"{GT_PATH.name:24s} NOT valid UTF-8 (first error at byte {e.start})")
    print("NOTE: non-ASCII in *test* files is where French names/addresses would show up.")

    # ---- 7. exact-match rates on true pairs --------------------------------
    hdr("7) EXACT-MATCH RATES ON KNOWN TRUE PAIRS (sampled)")
    agg = defaultdict(Counter)  # key: country label or '__ALL__'
    skipped = 0
    for s1, m in exact_sample:
        r1, r2 = captured.get(s1), captured.get(m)
        if not r1 or not r2:
            skipped += 1
            continue
        for key in ((r1[2] or "(empty)").strip() or "(empty)", "__ALL__"):
            a = agg[key]
            a["pairs"] += 1
            if r1[0].strip() and r2[0].strip():
                a["name_den"] += 1
                if r1[0] == r2[0]:
                    a["name_raw"] += 1
                if norm_text(r1[0]) == norm_text(r2[0]):
                    a["name_norm"] += 1
            if r1[1].strip() and r2[1].strip():
                a["addr_den"] += 1
                if r1[1] == r2[1]:
                    a["addr_raw"] += 1
                if norm_text(r1[1]) == norm_text(r2[1]):
                    a["addr_norm"] += 1
    if skipped:
        print(f"NOTE: {skipped} sampled pairs skipped (S1 side or match side not found "
              f"in the streamed files"
              f"{' — expected with --limit' if args.limit else ' — investigate!'}).\n")
    hdr("     exact-match rates, all countries")
    a = agg.get("__ALL__")
    if a:
        print(f"true pairs evaluated: {a['pairs']:,}")
        print(f"  name raw-equal : {pct(a['name_raw'], a['name_den'])}  "
              f"({a['name_raw']:,}/{a['name_den']:,} pairs where both names non-empty)")
        print(f"  name norm-equal: {pct(a['name_norm'], a['name_den'])}")
        print(f"  addr raw-equal : {pct(a['addr_raw'], a['addr_den'])}  "
              f"({a['addr_raw']:,}/{a['addr_den']:,} pairs where both addrs non-empty)")
        print(f"  addr norm-equal: {pct(a['addr_norm'], a['addr_den'])}   (bonus)")
    else:
        print("no evaluable pairs")
    print("\n  by S1 country:")
    for key in sorted(k for k in agg if k != "__ALL__"):
        a = agg[key]
        print(f"    {key:12s} n={a['pairs']:<7,d} name_raw={pct(a['name_raw'], a['name_den']):>6s} "
              f"name_norm={pct(a['name_norm'], a['name_den']):>6s} "
              f"addr_raw={pct(a['addr_raw'], a['addr_den']):>6s}")

    # ---- 8. sample true pairs ----------------------------------------------
    hdr(f"8) SAMPLE TRUE MATCH PAIRS ({len(print_pairs)})")
    for s1, m in print_pairs:
        r1, r2 = captured.get(s1), captured.get(m)
        if not r1 or not r2:
            print(f"[{s1}] <-> [{m}]   (row not captured"
                  f"{' — use full run, no --limit' if args.limit else ''})")
            continue
        print(f"[{s1}] name={trunc(r1[0])!r}  addr={trunc(r1[1])!r}  country={r1[2]}")
        print(f"  <-> [{m}] name={trunc(r2[0])!r}  addr={trunc(r2[1])!r}  country={r2[2]}")

    # ---- 9. sample near-miss non-pairs -------------------------------------
    hdr(f"9) SAMPLE NEAR-MISS NON-PAIRS (same block, NOT in GT) — {len(near_miss)} in pool")
    reason_counts = Counter(r for r, *_ in near_miss)
    print(f"pool by reason: {dict(reason_counts)}  (cap={NEAR_MISS_CAP})")
    if not near_miss:
        print("none found"
              + (" — expected with --limit; run without --limit for this section"
                 if args.limit else " — INVESTIGATE (blocking would generate zero "
                 "candidates for seeds)"))
    else:
        for reason, s1, other, s1_row in rng.sample(near_miss, min(args.miss_print, len(near_miss))):
            oid, oname, oaddr, ocountry = other
            if s1_row:
                print(f"[{s1}] name={trunc(s1_row[0])!r}  addr={trunc(s1_row[1])!r}")
            else:
                print(f"[{s1}] (S1 row not captured)")
            print(f"  <-> [{oid}] name={trunc(oname)!r}  addr={trunc(oaddr)!r}  "
                  f"country={ocountry}   reason={reason}")

    # ---- 10. GT vs source-file cross checks --------------------------------
    hdr("10) GROUND TRUTH vs SOURCE FILES (format sanity)")
    if args.limit:
        print("SKIPPED (unreliable under --limit) — re-run without --limit for:\n"
              "  * matched IDs missing from source files\n"
              "  * S1 entities with no GT row\n"
              "  * GT S1 IDs missing from train_source1")
    else:
        unseen = [m for m, seen in matched_ids_seen.items() if not seen]
        print(f"GT-referenced matched IDs NOT found in train_source2/3: {len(unseen)}")
        for m in unseen[:10]:
            print(f"  {m}")
        no_gt_row = sorted(train1_ids - set(gt))
        print(f"\ntrain_source1 entities with NO row in GT: {len(no_gt_row)}")
        for e in no_gt_row[:10]:
            print(f"  {e}")
        if no_gt_row:
            print("  -> are these treated as singletons by the scorer? "
                  "(ambiguity to resolve; GT says 'one row per S1')")
        phantom = sorted(set(gt) - train1_ids)
        print(f"\nGT source1 IDs NOT present in train_source1: {len(phantom)}")
        for e in phantom[:10]:
            print(f"  {e}")

    # ---- 11. train vs test drift + France ---------------------------------
    hdr("11) TRAIN vs TEST STRUCTURAL DRIFT + FRANCE SUBSET")
    print(f"{'file':24s} {'rows':>10s}  countries")
    for split, path in SOURCE_FILES:
        st = file_stats[path.name]
        labels = ", ".join(f"{c or '(empty)'}={n:,}" for c, n in st["countries"].most_common())
        print(f"{path.name:24s} {st['rows']:10,d}  {labels}")
    print("\nheader identical across all files:",
          all(file_stats[p.name]["header_ok"] for _, p in SOURCE_FILES))
    fr_counts = {}
    for _, path in SOURCE_FILES:
        if path.name.startswith("test_"):
            st = file_stats[path.name]
            fr_counts[path.name] = sum(n for c, n in st["countries"].items()
                                       if looks_french(c))
    print(f"France-looking rows in test files: {fr_counts}")
    print(f"\nFrance sample rows ({len(france_samples)} shown):")
    for fname, eid, name, addr in france_samples:
        print(f"  {fname} {eid}: name={name!r} addr={addr!r}")
    if not france_samples:
        print("  none captured"
              + (" — expected with --limit" if args.limit else
                 " — check country labels in section 4"))

    # ---- footer -------------------------------------------------------------
    hdr("DONE")
    print(
        "Please paste this ENTIRE output back (stdout) so we can lock in the\n"
        "blocking strategy, feature set, and model choice against real numbers.\n"
        f"Total runtime: {time.time() - t_start:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
