#!/usr/bin/env python3
"""normalize.py — text normalization, ID utilities, dataset streaming and caching.

Vendored verbatim from try1 (only path constants changed to try2).
Foundation for the try2 pipeline. Phase-0 scripts (id_check, count_families,
build_pool, analyze_misses) and later stages import from here.

Cache design
------------
Building normalized columns once per source file and storing them as fixed-width
``.npy`` memmapped arrays gives every later stage cheap sequential reads and
O(1) random access without ever holding a whole TSV (or a whole cache) in RAM.

Per-row columns written by :func:`ensure_cache`::

    id               int64   numeric part of entity_id (prefix tracked in meta)
    name_core        S90     normalized, legal-suffix-stripped name tokens
    name_sorted      S90     token-sorted core name
    name_suffixes    S32     legal-suffix tokens (space joined)
    name_initials    S10     first letters of core tokens
    name_phon        S12     crude phonetic key of first+last core token
    name_translit    S90     token-sorted core, transliterated to Latin
                             (== name_sorted for Latin-script rows / on failure)
    name_script      u1      script code (meta maps code -> label)
    addr_norm        S140    normalized address string
    addr_tokens      S160    normalized address tokens, space joined
    houseno          S12     first house-number-like token ('' if none)
    postal           S10     trailing 5-6 digit token ('' if none)
    state            S10     canonical state token ('' if not found)
    country          u1      country code (meta maps code -> label)
    name_nonascii    u1      1 if raw name had non-ASCII chars
    addr_nonascii    u1      1 if raw address had non-ASCII chars
"""

from __future__ import annotations

import os as _os, sys as _sys  # un-shadow stdlib 'inspect' (this dir has inspect.py)
_src_d = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0] or _os.getcwd()) == _src_d:
    _sys.path.pop(0)
    _sys.path.append(_src_d)


import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
TRY2_DIR = SRC_DIR.parent                      # experiments/try2
REPO_ROOT = TRY2_DIR.parents[1]                # repo root
DEFAULT_DATASET = REPO_ROOT / "student_resource" / "dataset"
DEFAULT_CACHE = TRY2_DIR / "cache"

# Cache layout: (width, dtype) per column.
CACHE_COLS: Dict[str, Tuple[int, str]] = {
    "id": (0, "int64"),
    "name_core": (90, "S"),
    "name_sorted": (90, "S"),
    "name_suffixes": (32, "S"),
    "name_initials": (10, "S"),
    "name_phon": (32, "S"),
    "name_translit": (90, "S"),
    "name_script": (0, "u1"),
    "addr_norm": (140, "S"),
    "addr_tokens": (160, "S"),
    "houseno": (12, "S"),
    "postal": (10, "S"),
    "state": (20, "S"),
    "country": (0, "u1"),
    "name_nonascii": (0, "u1"),
    "addr_nonascii": (0, "u1"),
}
NPY_DTYPES = {
    "int64": np.int64,
    "u1": np.uint8,
}

# --------------------------------------------------------------------------
# Lexicons (hand-written general knowledge — NOT external data lookup)
# --------------------------------------------------------------------------
SUFFIXES = {
    # US / UK / international
    "inc", "incorporated", "corp", "corporation", "llc", "llp", "lp", "ltd",
    "limited", "pvt", "private", "plc", "co", "company", "gmbh", "ag", "sa",
    "sas", "srl", "spa", "oy", "ab", "as", "bv", "nv", "pt", "pte", "kg",
    "kgaa", "ou", "te", "trust", "society", "association", "assoc",
    "partnership", "enterprises", "enterprise", "holdings", "holding",
    # India-specific
    "venture", "ventures",  # often boilerplate; kept out of core matching
    # France-specific
    "sasu", "eurl", "snc", "sci", "sc", "eirl", "selarl", "sel",
    # generic
    "firm", "group",  # NOTE: 'group'/'center' etc. left IN core on purpose
}
# Tokens we strip only as *address* noise, never as name tokens.
ADDR_STOP_TOKENS = {"the", "of", "and"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
US_STATE_ABBRS = set(US_STATES.values()) | {"dc"}

INDIA_STATES = {
    "andhra pradesh": "andhra pradesh", "arunachal pradesh": "arunachal pradesh",
    "assam": "assam", "bihar": "bihar", "chhattisgarh": "chhattisgarh",
    "goa": "goa", "gujarat": "gujarat", "haryana": "haryana",
    "himachal pradesh": "himachal pradesh", "jharkhand": "jharkhand",
    "karnataka": "karnataka", "kerala": "kerala", "madhya pradesh": "madhya pradesh",
    "maharashtra": "maharashtra", "manipur": "manipur", "meghalaya": "meghalaya",
    "mizoram": "mizoram", "nagaland": "nagaland", "odisha": "odisha",
    "orissa": "odisha", "punjab": "punjab", "rajasthan": "rajasthan",
    "sikkim": "sikkim", "tamil nadu": "tamil nadu", "telangana": "telangana",
    "tripura": "tripura", "uttar pradesh": "uttar pradesh",
    "uttarakhand": "uttarakhand", "west bengal": "west bengal",
    "delhi": "delhi", "jammu": "jammu", "kashmir": "kashmir",
    "puducherry": "puducherry", "pondicherry": "puducherry",
}
# Two-word names must be matched before single words; build a lookup of both.
STATE_LOOKUP: Dict[str, str] = {}
for _k, _v in {**US_STATES, **INDIA_STATES}.items():
    STATE_LOOKUP[_k] = _v
for _k, _v in list(US_STATES.items()):
    STATE_LOOKUP[_k.replace(" ", "")] = _v          # "newyork"
for _a in US_STATE_ABBRS:
    STATE_LOOKUP[_a] = _a
# single-word india states as themselves
for _w in ("assam", "bihar", "goa", "gujarat", "haryana", "jharkhand",
           "karnataka", "kerala", "manipur", "meghalaya", "mizoram",
           "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
           "telangana", "tripura", "delhi"):
    STATE_LOOKUP[_w] = _w

# Address-only abbreviation expansion (street types). 'st'/'ave' etc. are
# near-unambiguous inside addresses; names are NOT expanded ('St' may be Saint).
STREET_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "cir": "circle", "hwy": "highway", "pkwy": "parkway", "pl": "place",
    "sq": "square", "ter": "terrace", "trl": "trail", "xing": "crossing",
    "pkwy2": "parkway", "byp": "bypass", "expy": "expressway", "fwy": "freeway",
    "rte": "route", "mt": "mount", "ft": "fort", "hills": "hills",
}

# Name tokens too generic to form canopy keys on their own. (Bucket-size caps
# would catch these anyway; this just saves work.)
STOP_NAME_TOKENS = {"the", "and", "of", "for", "a", "an", "in", "on", "at"}

# Script codes -> labels (u1 column).
SCRIPT_LABELS = {
    0: "latin", 1: "devanagari", 2: "bengali", 3: "gurmukhi", 4: "gujarati",
    5: "oriya", 6: "tamil", 7: "telugu", 8: "kannada", 9: "malayalam",
    10: "thai", 11: "arabic", 12: "lao", 13: "myanmar", 14: "khmer",
    15: "japanese", 16: "chinese", 17: "korean", 255: "unknown",
}
SCRIPT_CODES = {v: k for k, v in SCRIPT_LABELS.items()}
_SCRIPT_RANGES = [
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"), (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"), (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"), (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"), (0x0E00, 0x0E7F, "thai"),
    (0x0600, 0x06FF, "arabic"), (0x0E80, 0x0EFF, "lao"),
    (0x1000, 0x109F, "myanmar"), (0x1780, 0x17FF, "khmer"),
    (0x3040, 0x30FF, "japanese"), (0x4E00, 0x9FFF, "chinese"),
    (0xAC00, 0xD7AF, "korean"),
]

COUNTRY_LABELS = {0: "US", 1: "India", 2: "France", 255: "unknown"}
COUNTRY_CODES = {v: k for k, v in COUNTRY_LABELS.items()}

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d{1,6}(?:-\d{1,6})?")
_POSTAL_RE = re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)|(?<!\d)\d{6}(?!\d)")
_ID_RE = re.compile(r"^(S[123])-(\d+)$")
S3_OFFSET = 1 << 31

def h64(s: str) -> int:
    """Deterministic 63-bit hash (stable across runs — unlike PYTHONHASHSEED).

    Masked to 63 bits so the value always fits a positive int64.
    """
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
        "little") & 0x7FFFFFFFFFFFFFFF

def parse_id(eid: str) -> Tuple[str, int]:
    """'S2-166376419' -> ('S2', 166376419). Raises on unexpected formats."""
    m = _ID_RE.match(eid.strip())
    if not m:
        raise ValueError(f"unexpected entity_id format: {eid!r}")
    return m.group(1), int(m.group(2))

def encode_other(prefix: str, n: int) -> int:
    """S2-n -> n ; S3-n -> n + 2**31 (single int64 space for S2+S3 ids)."""
    if prefix == "S2":
        return n
    if prefix == "S3":
        return n + S3_OFFSET
    raise ValueError(f"not a Source-2/3 id: {prefix}-{n}")

def decode_other(code: int) -> str:
    if code >= S3_OFFSET:
        return f"S3-{code - S3_OFFSET}"
    return f"S2-{code}"

def fix_mojibake(s: str) -> str:
    """Light repair for the most common double-encoding artefacts.

    Most mojibake differences cancel anyway: both sides of a pair pass through
    the same normalizer, and de-accent + punctuation-stripping absorbs the rest
    ('VspâS' and \"Vsp's\" both -> 'vsps').
    """
    for pat, rep in (
        ("â€™", "'"), ("â€œ", '"'), ("â€\x9d", '"'),
        ("â€“", "-"), ("â€”", "-"), ("Â—", "-"), ("Â–", "-"),
        ("Ã¢", "-"),
    ):
        if pat in s:
            s = s.replace(pat, rep)
    return s

def deaccent(s: str) -> str:
    """NFD + drop combining marks + fold a few special letters to ASCII."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return (s.replace("ß", "ss").replace("æ", "ae").replace("œ", "oe")
             .replace("Æ", "AE").replace("ø", "o").replace("Ø", "O")
             .replace("đ", "d").replace("ł", "l").replace("ð", "d"))

def detect_script(s: str) -> str:
    """Return script label of the first non-Latin letter found ('latin' if none).

    Latin includes Latin-1 supplement + Latin Extended (French accents).
    """
    for ch in s:
        o = ord(ch)
        if o < 0x180:                     # ASCII + Latin-1 supplement
            continue
        if 0x180 <= o <= 0x24F:           # Latin Extended-A/B
            continue
        for lo, hi, label in _SCRIPT_RANGES:
            if lo <= o <= hi:
                return label
        if ch.isalpha():
            return "unknown"
    return "latin"

# ---- transliteration (optional dependency; graceful degradation) ----------
_SANSCRIPT = None  # False = tried and failed; None = not tried; else module

def _get_sanscript():
    global _SANSCRIPT
    if _SANSCRIPT is None:
        try:
            from indic_transliteration import sanscript  # type: ignore
            _SANSCRIPT = sanscript
        except Exception:
            _SANSCRIPT = False
    return _SANSCRIPT

_SCRIPT_TO_SANSCRIPT = {
    "devanagari": "DEVANAGARI", "bengali": "BENGALI", "gurmukhi": "GURMUKHI",
    "gujarati": "GUJARATI", "oriya": "ORIYA", "tamil": "TAMIL",
    "telugu": "TELUGU", "kannada": "KANNADA", "malayalam": "MALAYALAM",
}

def transliterate_to_latin(s: str, script: str) -> Optional[str]:
    """Transliterate an Indic-script string to Latin (ITRANS scheme).

    Returns None when the script is unsupported or the optional library is
    unavailable — callers fall back to using the raw string.
    """
    if script == "latin":
        return s
    ss = _get_sanscript()
    if not ss:
        return None
    src = _SCRIPT_TO_SANSCRIPT.get(script)
    if src is None:
        return None
    try:
        out = ss.transliterate(s, getattr(ss, src), ss.ITRANS)
        return out
    except Exception:
        return None

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def _basic(s: str) -> str:
    """mojibake repair -> NFKC -> casefold -> de-accent -> &='and' -> punct."""
    s = fix_mojibake(s)
    s = unicodedata.normalize("NFKC", s).casefold()
    s = deaccent(s)
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()

class NameInfo(NamedTuple):
    full: str          # normalized, order kept, suffixes kept
    core: str          # suffixes stripped, order kept
    sorted: str        # suffixes stripped, token sorted
    suffixes: str      # suffix tokens only
    initials: str
    phon: str
    script: str
    nonascii: int

def phon_key(tok: str) -> str:
    """Crude metaphone-ish key: first letter + consonant skeleton."""
    letters = [c for c in tok.upper() if "A" <= c <= "Z"]
    if not letters:
        return ""
    out = [letters[0]]
    for c in letters[1:]:
        if c in "AEIOU":
            continue
        if out[-1] == c:
            continue
        out.append(c)
    return "".join(out)

def norm_name(raw: str) -> NameInfo:
    nonascii = int(not raw.isascii())
    full = _basic(raw)
    toks = [t for t in full.split() if t]
    if not toks:
        return NameInfo("", "", "", "", "", "",
                        detect_script(raw), nonascii)
    suf = [t for t in toks if t in SUFFIXES]
    core = [t for t in toks if t not in SUFFIXES] or toks
    core_s = " ".join(core)
    sorted_s = " ".join(sorted(core))
    initials = "".join(t[0] for t in core if t and t[0].isalpha())
    if len(core) >= 2:
        phon = phon_key(core[0]) + " " + phon_key(core[-1])
    else:
        phon = phon_key(core[0])
    script = detect_script(raw)
    return NameInfo(full, core_s, sorted_s, " ".join(sorted(set(suf))),
                    initials, phon, script, nonascii)

class AddrInfo(NamedTuple):
    norm: str
    tokens: str
    houseno: str
    postal: str
    state: str
    nonascii: int

def norm_address(raw: str) -> AddrInfo:
    nonascii = int(not raw.isascii())
    if not raw.strip():
        return AddrInfo("", "", "", "", "", nonascii)
    work = fix_mojibake(raw)
    work = unicodedata.normalize("NFKC", work).casefold()
    work = deaccent(work)
    work = work.replace("&", " and ")

    # postal: trailing 5-6 digit token (US ZIP / India PIN / FR code postal)
    postal = ""
    m_post = _POSTAL_RE.search(work)
    if m_post:
        # prefer a token at/нear the end; otherwise still accept
        postal = m_post.group(0).replace("-", "")

    # house number: first short numeric token that is not the postal
    houseno = ""
    for m in _NUM_RE.finditer(work):
        cand = m.group(0)
        if cand.replace("-", "") == postal:
            continue
        # skip obvious postal-like runs sitting at the tail
        if m.end() >= len(work) - 4 and len(cand) >= 5:
            continue
        houseno = cand
        break

    # address tokens (punct -> space), street-abbrev expanded
    toks = []
    for t in _PUNCT_RE.sub(" ", work).split():
        t = STREET_ABBR.get(t, t) if len(t) <= 5 else t
        if t:
            toks.append(t)
    # state: two-word then one-word lookup over original-ish token stream
    state = ""
    for i in range(len(toks) - 1):
        pair = toks[i] + " " + toks[i + 1]
        if pair in STATE_LOOKUP:
            state = STATE_LOOKUP[pair]
            break
    if not state:
        for t in toks:
            if t in STATE_LOOKUP:
                state = STATE_LOOKUP[t]
                break
    norm = " ".join(toks)
    return AddrInfo(norm, norm, houseno, postal, state, nonascii)

class RowNorm(NamedTuple):
    """One fully-normalized record (cache row)."""
    id: int
    name_core: str
    name_sorted: str
    name_suffixes: str
    name_initials: str
    name_phon: str
    name_translit: str
    name_script: int
    addr_norm: str
    addr_tokens: str
    houseno: str
    postal: str
    state: str
    country: int
    name_nonascii: int
    addr_nonascii: int

def normalize_row(id_num: int, name: str, addr: str, country: str) -> RowNorm:
    ni = norm_name(name)
    ai = norm_address(addr)
    script = ni.script
    translit_sorted = ni.sorted
    if script != "latin":
        t = transliterate_to_latin(name, script)
        if t:
            ti = norm_name(t)           # re-normalize the Latin rendering
            translit_sorted = ti.sorted or ni.sorted
    return RowNorm(
        id=id_num,
        name_core=ni.core,
        name_sorted=ni.sorted,
        name_suffixes=ni.suffixes,
        name_initials=ni.initials,
        name_phon=ni.phon,
        name_translit=translit_sorted,
        name_script=SCRIPT_CODES.get(script, 255),
        addr_norm=ai.norm,
        addr_tokens=ai.tokens,
        houseno=ai.houseno,
        postal=ai.postal,
        state=ai.state,
        country=COUNTRY_CODES.get(country.strip(), 255),
        name_nonascii=ni.nonascii,
        addr_nonascii=ai.nonascii,
    )

# --------------------------------------------------------------------------
# Streaming raw source files
# --------------------------------------------------------------------------
class RawRow(NamedTuple):
    id_num: int
    prefix: str
    name: str
    addr: str
    country: str

def stream_source(path: Path, limit: Optional[int] = None) -> Iterator[RawRow]:
    """Yield raw rows from a source TSV (utf-8, errors replaced on bad bytes)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)  # header
        n = 0
        for line in f:
            if limit is not None and n >= limit:
                return
            parts = line.rstrip("\n").rstrip("\r").split("\t", 3)
            while len(parts) < 4:
                parts.append("")
            try:
                prefix, num = parse_id(parts[0])
            except ValueError:
                continue
            n += 1
            yield RawRow(num, prefix, parts[1], parts[2], parts[3])

def stream_ground_truth(path: Path) -> Iterator[Tuple[int, List[int]]]:
    """Yield (s1_num, [other_encoded...]) — encoded via encode_other()."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if not parts or not parts[0]:
                continue
            try:
                _, s1 = parse_id(parts[0])
            except ValueError:
                continue
            ids: List[int] = []
            if len(parts) > 1 and parts[1].strip():
                for tok in parts[1].split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    try:
                        pfx, num = parse_id(tok)
                        ids.append(encode_other(pfx, num))
                    except ValueError:
                        continue
            yield s1, ids

# --------------------------------------------------------------------------
# Cache build / load
# --------------------------------------------------------------------------
def _trunc_bytes(s: str, width: int) -> bytes:
    b = s.encode("utf-8")
    if len(b) <= width:
        return b
    b = b[:width]
    while b:
        try:
            b.decode("utf-8")
            return b
        except UnicodeDecodeError:
            b = b[:-1]
    return b

def _col_path(cache_dir: Path, stem: str, col: str) -> Path:
    return cache_dir / f"{stem}__{col}.npy"

def _count_rows(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1

def ensure_cache(split: str, dataset: Path, cache_dir: Path,
                 limit: Optional[int] = None, force: bool = False) -> Dict[str, Path]:
    """Build (or reuse) per-source .npy caches. Returns {stem: path_prefix}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stems = [f"{split}_source1", f"{split}_source2", f"{split}_source3"]
    out: Dict[str, Path] = {}
    for stem in stems:
        src = dataset / split / f"{stem}.tsv"
        meta_path = cache_dir / f"{stem}__meta.json"
        current_widths = {c: w for c, (w, k) in CACHE_COLS.items() if k == "S"}
        if meta_path.is_file() and not force and limit is None:
            meta = json.loads(meta_path.read_text())
            if (meta.get("src_size") == src.stat().st_size
                    and meta.get("widths") == current_widths):
                out[stem] = cache_dir / stem
                continue
        if not src.is_file():
            raise FileNotFoundError(f"missing dataset file: {src}")
        n = _count_rows(src)
        if limit is not None:
            n = min(n, limit)
        print(f"[cache] building {stem} ({n:,} rows) ...", flush=True)
        arrays = {}
        for col, (width, kind) in CACHE_COLS.items():
            if kind == "S":
                dt = f"S{width}"
            else:
                dt = NPY_DTYPES[kind]
            arrays[col] = np.lib.format.open_memmap(
                _col_path(cache_dir, stem, col), mode="w+", dtype=dt, shape=(n,))
        trunc = {c: 0 for c, (w, k) in CACHE_COLS.items() if k == "S"}
        i = 0
        for row in stream_source(src, limit=limit):
            rn = normalize_row(row.id_num, row.name, row.addr, row.country)
            arrays["id"][i] = rn.id
            for col in ("name_core", "name_sorted", "name_suffixes",
                        "name_initials", "name_phon", "name_translit",
                        "addr_norm", "addr_tokens", "houseno", "postal", "state"):
                width = CACHE_COLS[col][0]
                val = getattr(rn, col)
                b = _trunc_bytes(val, width)
                if len(val.encode("utf-8")) > width:
                    trunc[col] += 1
                arrays[col][i] = b
            arrays["name_script"][i] = rn.name_script
            arrays["country"][i] = rn.country
            arrays["name_nonascii"][i] = rn.name_nonascii
            arrays["addr_nonascii"][i] = rn.addr_nonascii
            i += 1
        for a in arrays.values():
            a.flush()
            del a
        meta = {
            "rows": i,
            "src_size": src.stat().st_size,
            "widths": current_widths,
            "truncated": {k: v for k, v in trunc.items() if v},
            "stem": stem,
        }
        meta_path.write_text(json.dumps(meta))
        bad = {k: v for k, v in meta["truncated"].items() if v}
        if bad:
            print(f"[cache]   note: width-truncated columns in {stem}: {bad}")
        print(f"[cache]   done ({i:,} rows)", flush=True)
        out[stem] = cache_dir / stem
    return out

class SourceCache:
    """Memmapped normalized columns for one source file."""

    def __init__(self, prefix: Path, meta: Optional[dict] = None):
        self.prefix = prefix
        stem = prefix.name
        self.meta = meta or json.loads(
            (prefix.parent / f"{stem}__meta.json").read_text())
        self.arrays: Dict[str, np.ndarray] = {}
        for col, (width, kind) in CACHE_COLS.items():
            dt = f"S{width}" if kind == "S" else NPY_DTYPES[kind]
            self.arrays[col] = np.load(_col_path(prefix.parent, stem, col),
                                       mmap_mode="r")
        self.rows = int(self.meta["rows"])
        self.ids = np.asarray(self.arrays["id"])
        # sorted-id index for entity-id -> row lookup
        self._order = np.argsort(self.ids, kind="stable")
        self._sorted_ids = self.ids[self._order]

    def row_of(self, id_nums: np.ndarray) -> np.ndarray:
        """entity id -> row index (row == -1 when absent)."""
        pos = np.searchsorted(self._sorted_ids, id_nums)
        pos = np.clip(pos, 0, len(self._sorted_ids) - 1)
        rows = self._order[pos]
        ok = self._sorted_ids[pos] == id_nums
        return np.where(ok, rows, -1)

    def get(self, col: str, rows: np.ndarray) -> np.ndarray:
        return self.arrays[col][rows]

def decode_str(a: np.ndarray) -> np.ndarray:
    """bytes/`S` array -> unicode str array (utf-8, replacement on damage)."""
    return np.char.decode(a.astype("S" + str(a.dtype.itemsize)), "utf-8",
                          errors="replace")

def add_common_args(ap) -> None:
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help=f"dataset dir (default: {DEFAULT_DATASET})")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                    help=f"cache dir (default: {DEFAULT_CACHE})")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows per source file (smoke tests only)")

def resolve_split_stems(split: str, dataset: Path, cache_dir: Path,
                        limit: Optional[int] = None, force: bool = False):
    ensure_cache(split, dataset, cache_dir, limit=limit, force=force)
    return {i: SourceCache(cache_dir / f"{split}_source{i}") for i in (1, 2, 3)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build/inspect the normalization cache.")
    add_common_args(ap)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    caches = resolve_split_stems(args.split, args.dataset, args.cache_dir,
                                 limit=args.limit, force=args.force)
    for i, c in caches.items():
        print(f"source{i}: {c.rows:,} rows, "
              f"nonascii names={int(np.sum(c.arrays['name_nonascii'])):,}")
    print("cache OK")
