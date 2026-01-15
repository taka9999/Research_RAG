#!/usr/bin/env python3
"""
Suggest EXPAND candidates from chunks.jsonl.

Outputs JSON with:
- abbrev_pairs_by_topic: list of {"abbr":..., "long":..., "count":..., "examples":[...]} per topic
- ngrams_by_topic: list of {"ngram":..., "n":2/3, "count":...} per topic

This is meant to be *semi-automatic*: you review the suggestions and then manually
curate them into retriever_bm25.EXPAND or a topic-specific EXPAND map.

Usage:
  python rag/expand_suggest.py --chunks index/chunks.jsonl --out rag/expand_suggestions.json --top_abbr 80 --top_ngram 200
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Very small stopword list (English). Extend if needed.
STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","by","as","at","from","that","this","these","those",
    "is","are","was","were","be","been","being","it","its","we","our","you","your","they","their","i","me","my",
    "not","no","yes","can","could","may","might","should","would","will","do","does","did","done",
    "into","over","under","between","within","without","than","then","there","here","such","also",
}

# Pattern for "Long Form (ABBR)"
# ABBR is 2-10 chars, mostly uppercase letters/digits/hyphen.
RE_LONG_ABBR = re.compile(
    r"""(?P<long>[A-Za-z][A-Za-z0-9\- ]{6,120}?)\s*\(\s*(?P<abbr>[A-Z][A-Z0-9\-]{1,9})\s*\)"""
)

# Optional reverse pattern "ABBR (Long Form)"
RE_ABBR_LONG = re.compile(
    r"""\b(?P<abbr>[A-Z][A-Z0-9\-]{1,9})\s*\(\s*(?P<long>[A-Za-z][A-Za-z0-9\- ]{6,120}?)\s*\)"""
)

# Tokenizer: keeps letters/digits; splits on other chars.
RE_TOKEN = re.compile(r"[A-Za-z0-9]+")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def normalize_long(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # Trim trailing punctuation
    s = s.strip(" ,.;:")
    return s


def normalize_abbr(s: str) -> str:
    return s.strip()


def extract_topics(rec: Dict[str, Any]) -> List[str]:
    meta = rec.get("meta") or {}
    t = meta.get("topic")
    if t is None:
        return ["__unknown__"]
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [str(x) for x in t] if t else ["__unknown__"]
    return ["__unknown__"]


def extract_text(rec: Dict[str, Any]) -> str:
    # Common keys: "text" or "chunk" etc.
    for k in ("text", "content", "chunk_text"):
        if k in rec and isinstance(rec[k], str):
            return rec[k]
    return ""


def add_example(examples: List[str], s: str, limit: int = 3) -> None:
    if len(examples) >= limit:
        return
    s = s.strip()
    if not s:
        return
    if s not in examples:
        examples.append(s[:220])


def abbreviation_suggestions(records: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Counter], Dict[Tuple[str,str,str], List[str]]]:
    """
    Returns:
      counts_by_topic: topic -> Counter[(abbr, long)]
      examples: (topic, abbr, long) -> [example snippets]
    """
    counts_by_topic: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    for rec in records:
        text = extract_text(rec)
        if not text:
            continue
        topics = extract_topics(rec)
        # Search both patterns; prefer Long(ABBR) first.
        matches = list(RE_LONG_ABBR.finditer(text)) + list(RE_ABBR_LONG.finditer(text))
        for m in matches:
            abbr = normalize_abbr(m.group("abbr"))
            long = normalize_long(m.group("long"))
            # Filter: long should contain at least 2 words
            if len(long.split()) < 2:
                continue
            # Filter: avoid cases where long is mostly uppercase (often section headers)
            if sum(c.isupper() for c in long) > 0.6 * max(1, len(long)):
                continue
            for topic in topics:
                counts_by_topic[topic][(abbr, long)] += 1
                key = (topic, abbr, long)
                # capture a bit of surrounding context as example
                span = m.span()
                lo = max(0, span[0] - 60)
                hi = min(len(text), span[1] + 60)
                add_example(examples[key], text[lo:hi])
    return counts_by_topic, examples


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in RE_TOKEN.findall(text)]
    toks = [t for t in toks if t not in STOPWORDS and not t.isdigit()]
    # Drop very short tokens
    toks = [t for t in toks if len(t) >= 2]
    return toks


def ngram_suggestions(records: Iterable[Dict[str, Any]], *, n_values=(2,3)) -> Dict[str, Dict[int, Counter]]:
    """
    Returns: topic -> n -> Counter[ngram_str]
    """
    out: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for rec in records:
        text = extract_text(rec)
        if not text:
            continue
        topics = extract_topics(rec)
        toks = tokenize(text)
        if len(toks) < 3:
            continue
        for n in n_values:
            if len(toks) < n:
                continue
            grams = (" ".join(toks[i:i+n]) for i in range(0, len(toks)-n+1))
            for g in grams:
                # Filter: avoid grams that include very generic words (still can be noisy)
                if any(w in {"model","models","method","methods","result","results","data","analysis"} for w in g.split()):
                    continue
                for topic in topics:
                    out[topic][n][g] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, required=True, help="Path to chunks.jsonl")
    ap.add_argument("--out", type=str, required=True, help="Output JSON path, e.g., expand_suggestions.json")
    ap.add_argument("--top_abbr", type=int, default=80, help="Top abbreviation pairs per topic")
    ap.add_argument("--top_ngram", type=int, default=200, help="Top n-grams per topic (combined for n=2,3)")
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    rows = list(iter_jsonl(chunks_path))

    # Abbreviations
    abbr_counts, abbr_examples = abbreviation_suggestions(rows)
    abbrev_pairs_by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for topic, ctr in abbr_counts.items():
        items = []
        for (abbr, long), cnt in ctr.most_common(args.top_abbr):
            key = (topic, abbr, long)
            items.append({
                "abbr": abbr,
                "long": long,
                "count": cnt,
                "examples": abbr_examples.get(key, [])[:3],
            })
        abbrev_pairs_by_topic[topic] = items

    # N-grams
    ng = ngram_suggestions(rows, n_values=(2,3))
    ngrams_by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for topic, by_n in ng.items():
        merged = []
        for n, ctr in by_n.items():
            for g, cnt in ctr.items():
                merged.append((cnt, n, g))
        merged.sort(reverse=True)
        merged = merged[:args.top_ngram]
        ngrams_by_topic[topic] = [{"ngram": g, "n": n, "count": cnt} for (cnt, n, g) in merged]

    out = {
        "source_chunks": str(chunks_path),
        "abbrev_pairs_by_topic": abbrev_pairs_by_topic,
        "ngrams_by_topic": ngrams_by_topic,
        "notes": [
            "This file is for review. Add selected items into your EXPAND dictionary.",
            "Abbrev extraction looks for patterns like 'Long Form (ABBR)' and 'ABBR (Long Form)'.",
            "N-grams are simple frequency counts; expect noise—treat as a candidate list.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote suggestions -> {out_path}")


if __name__ == "__main__":
    main()
