#!/usr/bin/env python3
"""enable_translit.py — Phase 1, Step 3: activate transliteration for real.

License gate (checked before shipping): indic-transliteration is MIT
(PyPI classifier "OSI Approved :: MIT License", verified 2026-09).

What it does:
  1. prints the detected backend (or install instructions if absent)
  2. calls ensure_cache for the requested split — the meta stamp
     `translit_backend` was added in Phase 1, so ANY cache whose stamp
     differs from the currently-installed backend is rebuilt automatically
     (other caches are left alone)
  3. verifies the result: shows a non-Latin row's name_sorted vs the now-
     Latin name_translit

Run AFTER trace_misses (tracer must see the pool's key generation) and
BEFORE build_pool --rescore. Train ~3-4 min, test ~3-4 min.
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

import numpy as np

import normalize as N


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    N.add_common_args(ap)
    ap.add_argument("--split", default="train",
                    choices=["train", "test", "all"])
    args = ap.parse_args()

    backend = N._translit_backend()
    print(f"[translit] backend: {backend}")
    if backend == "none":
        raise SystemExit(
            "[translit] library NOT available. Install (MIT):\n"
            "    pip install indic-transliteration==2.3.82\n"
            "  then re-run this script. No cache was touched.")
    print("[translit] license: MIT (verified against PyPI metadata)",
          flush=True)

    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        t0 = time.time()
        # validity check inside ensure_cache now includes translit_backend,
        # so mismatched stamps rebuild; matching stamps are reused.
        N.ensure_cache(split, args.dataset, args.cache_dir,
                       limit=args.limit, force=False)
        meta_p = args.cache_dir / f"{split}_source2__meta.json"
        stamp = json.loads(meta_p.read_text()).get("translit_backend")
        print(f"[translit] {split}: meta stamp = {stamp} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- verification demo: first non-Latin S2 row ----
    c2 = N.SourceCache(args.cache_dir / f"{split}_source2")
    sc = np.asarray(c2.arrays["name_script"])
    nz = np.flatnonzero(sc != 0)
    if len(nz):
        r = int(nz[0])
        ns = c2.arrays["name_sorted"][r].decode("utf-8", "replace")
        tr = c2.arrays["name_translit"][r].decode("utf-8", "replace")
        print(f"\n[translit] verify row {r}:")
        print(f"  name_sorted  : {ns}")
        print(f"  name_translit: {tr}")
        if tr == ns:
            print("  WARNING: translit identical to native — backend may "
                  "be failing silently")
        else:
            print("  OK — Latin rendering differs from native script")
    else:
        print("[translit] no non-Latin rows in source2 (unexpected)")
    print("[translit] done — next: python src/build_pool.py --rescore ...")


if __name__ == "__main__":
    main()
