#!/usr/bin/env python3
"""
Interactive EXPAND curator for retriever_bm25.py.

- Reads expand_suggestions.json produced by expand_suggest.py
- Lets you accept/reject abbreviation pairs (Long Form (ABBR))
- Optionally curate n-grams (topic-specific) by manually attaching expansions
- Outputs a unified diff patch that updates EXPAND in retriever_bm25.py

Usage:
  python rag/expand_curate.py \
    --suggest rag/expand_suggestions.json \
    --bm25 rag/retriever_bm25.py \
    --out rag/expand_patch.diff

Optional:
  --topic reinforcement_learning
  --limit_abbr 60
  --limit_ngram 40
  --ngrams        # enable n-gram curation (asks for expansions)
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_expand_block(src: str) -> Tuple[Dict[str, List[str]], int, int, str]:
    """
    Returns (expand_dict, start_idx, end_idx, block_text)
    where [start_idx:end_idx] in src corresponds to the dict literal including braces.
    """
    m = re.search(r"^EXPAND\s*=\s*\{", src, flags=re.M)
    if not m:
        raise ValueError("Could not find 'EXPAND = {' in bm25 file.")
    start = m.end() - 1  # position of '{'

    # Find matching closing brace for the dict literal.
    i = start
    depth = 0
    in_str = False
    str_ch = ""
    escape = False
    while i < len(src):
        ch = src[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == str_ch:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                str_ch = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    block = src[start:end]
                    # Parse dict literal safely.
                    expand = ast.literal_eval(block)
                    if not isinstance(expand, dict):
                        raise ValueError("Parsed EXPAND is not a dict.")
                    # Normalize: keys must be str, values list[str]
                    out: Dict[str, List[str]] = {}
                    for k, v in expand.items():
                        if not isinstance(k, str):
                            continue
                        if isinstance(v, list):
                            vv = [str(x) for x in v]
                        else:
                            vv = [str(v)]
                        out[k] = vv
                    return out, start, end, block
        i += 1
    raise ValueError("Could not find matching closing '}' for EXPAND dict.")


def uniq_extend(existing: List[str], add: List[str]) -> List[str]:
    seen = {x.lower(): x for x in existing}
    for w in add:
        wl = w.lower()
        if wl not in seen:
            existing.append(w)
            seen[wl] = w
    return existing


def normalize_key(k: str) -> str:
    return re.sub(r"\s+", " ", k.strip().lower())


def format_expand_dict(d: Dict[str, List[str]]) -> str:
    """
    Pretty-print EXPAND dict with stable ordering.
    """
    # Group by rough themes: unknown first, then alpha.
    keys = sorted(d.keys(), key=lambda x: (x.startswith("__"), x))
    lines = ["{"]
    for k in keys:
        vals = d[k]
        # Keep values stable: preserve order, but sort within for determinism could be too aggressive.
        # We'll keep insertion order as already curated; only ensure no duplicates by lower.
        vv = []
        seen = set()
        for w in vals:
            wl = w.lower()
            if wl in seen:
                continue
            seen.add(wl)
            vv.append(w)

        # Render list
        items = ", ".join([repr(x) for x in vv])
        lines.append(f"    {repr(k)}: [{items}],")
    lines.append("}")
    return "\n".join(lines)


def prompt_yn(msg: str, default: str = "n") -> bool:
    default = default.lower()
    assert default in ("y", "n")
    suf = " [Y/n] " if default == "y" else " [y/N] "
    while True:
        ans = input(msg + suf).strip().lower()
        if not ans:
            ans = default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def prompt_list(msg: str) -> List[str]:
    ans = input(msg).strip()
    if not ans:
        return []
    # comma-separated
    parts = [p.strip() for p in ans.split(",")]
    return [p for p in parts if p]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suggest", type=str, required=True, help="expand_suggestions.json")
    ap.add_argument("--bm25", type=str, required=True, help="Path to retriever_bm25.py")
    ap.add_argument("--out", type=str, required=True, help="Output unified diff patch file")
    ap.add_argument("--topic", type=str, default="", help="Only curate a single topic")
    ap.add_argument("--limit_abbr", type=int, default=60, help="Max abbreviation pairs per topic to prompt")
    ap.add_argument("--limit_ngram", type=int, default=40, help="Max ngrams per topic to prompt (when --ngrams)")
    ap.add_argument("--ngrams", action="store_true", help="Enable interactive n-gram curation (asks for expansions)")
    args = ap.parse_args()

    suggest_path = Path(args.suggest)
    bm25_path = Path(args.bm25)
    out_path = Path(args.out)

    sugg = load_json(suggest_path)
    bm25_src = bm25_path.read_text(encoding="utf-8")

    expand, start, end, _ = extract_expand_block(bm25_src)
    expand_norm = {normalize_key(k): k for k in expand.keys()}  # normalized -> original key

    topics = sorted(set(list((sugg.get("abbrev_pairs_by_topic") or {}).keys()) + list((sugg.get("ngrams_by_topic") or {}).keys())))
    if args.topic:
        topics = [t for t in topics if t == args.topic]
        if not topics:
            raise SystemExit(f"No such topic in suggestions: {args.topic}")

    added_count = 0

    # --- Abbreviations ---
    abbr_by_topic: Dict[str, List[Dict[str, Any]]] = sugg.get("abbrev_pairs_by_topic") or {}
    for topic in topics:
        pairs = abbr_by_topic.get(topic) or []
        if not pairs:
            continue
        print("\n" + "="*80)
        print(f"Topic: {topic} | Abbreviation pairs (showing up to {args.limit_abbr})")
        print("="*80)
        for item in pairs[:args.limit_abbr]:
            abbr = str(item.get("abbr", "")).strip()
            long = str(item.get("long", "")).strip()
            cnt = int(item.get("count", 0) or 0)
            examples = item.get("examples") or []
            if not abbr or not long:
                continue

            print(f"\n{abbr}  <->  {long}   (count={cnt})")
            for ex in examples[:2]:
                print(f"  ex: ...{ex}...")

            if not prompt_yn("Add this pair bidirectionally?", default="y"):
                continue

            # Bidirectional entries:
            # key "ppo" -> ["Proximal Policy Optimization", ...]
            # key "proximal policy optimization" -> ["PPO", ...]
            k1 = normalize_key(abbr)
            v1 = [long]
            k2 = normalize_key(long)
            v2 = [abbr]

            # Merge into EXPAND (preserve original key casing if exists)
            for kk, vv in ((k1, v1), (k2, v2)):
                if kk in expand_norm:
                    orig = expand_norm[kk]
                    expand[orig] = uniq_extend(expand[orig], vv)
                else:
                    expand[kk] = vv
                    expand_norm[kk] = kk
                added_count += 1

            # Optional: allow extra synonyms
            extra = prompt_list("Optional: add extra expansions (comma-separated) for this ABBR key, or Enter to skip: ")
            if extra:
                kk = normalize_key(abbr)
                orig = expand_norm[kk]
                expand[orig] = uniq_extend(expand[orig], extra)

    # --- N-grams (optional) ---
    if args.ngrams:
        ngrams_by_topic: Dict[str, List[Dict[str, Any]]] = sugg.get("ngrams_by_topic") or {}
        for topic in topics:
            grams = ngrams_by_topic.get(topic) or []
            if not grams:
                continue
            print("\n" + "="*80)
            print(f"Topic: {topic} | N-grams (showing up to {args.limit_ngram})")
            print("="*80)
            print("For n-grams, you choose whether to add them as trigger-keys, and you must provide expansions.")
            print("Tip: use this to connect a phrase to canonical tokens (e.g., 'trust region' -> 'TRPO, KL').")

            for item in grams[:args.limit_ngram]:
                ngram = str(item.get("ngram","")).strip()
                cnt = int(item.get("count", 0) or 0)
                if not ngram:
                    continue
                print(f"\nngram: '{ngram}' (count={cnt})")
                if not prompt_yn("Add this n-gram as a trigger key?", default="n"):
                    continue
                expansions = prompt_list("Enter expansions to append when this n-gram appears (comma-separated): ")
                if not expansions:
                    print("  -> skipped (no expansions provided)")
                    continue
                kk = normalize_key(ngram)
                if kk in expand_norm:
                    orig = expand_norm[kk]
                    expand[orig] = uniq_extend(expand[orig], expansions)
                else:
                    expand[kk] = expansions
                    expand_norm[kk] = kk
                added_count += 1

    if added_count == 0:
        print("\n[INFO] No changes selected. No patch written.")
        return

    # Re-render EXPAND dict and build patch
    new_block = format_expand_dict(expand)
    new_src = bm25_src[:start] + new_block + bm25_src[end:]

    diff = difflib.unified_diff(
        bm25_src.splitlines(True),
        new_src.splitlines(True),
        fromfile=str(bm25_path),
        tofile=str(bm25_path),
        n=3,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(diff), encoding="utf-8")
    print(f"\n[OK] wrote patch -> {out_path}")
    print("Apply with:")
    print(f"  git apply {out_path}")


if __name__ == "__main__":
    main()
