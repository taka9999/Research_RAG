from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from retriever_tfidf import TfidfRetriever, Chunk
from retriever_bm25 import BM25Retriever


# -------------------------
# Dataset format
# -------------------------
# Expect a JSONL with items like:
# {
#   "id": "q001",
#   "question": "...",
#   "topic": "econometrics",              # optional
#   "gold_chunk_ids": ["paper::p3::c0"],  # optional (for retrieval eval)
#   "answer": ".... (chunk_id, p#) ..."   # optional (for faithfulness eval)
# }
#
# Notes:
# - If gold_chunk_ids missing, we skip recall@k for that item.
# - If answer missing, we skip faithfulness for that item.

# chunk_id にスペースが入っても拾う（カンマまでを chunk_id とみなす）
CITE_RE = re.compile(r"\(([^,]+?)\s*,\s*p(\d+)\)")


@dataclass
class Metrics:
    n_total: int = 0
    n_with_gold: int = 0
    n_with_answer: int = 0

    recall_at_k: float = float("nan")

    # Faithfulness-ish:
    citation_sentence_rate: float = float("nan")      # share of sentences that contain a citation
    bad_citation_rate: float = float("nan")           # citations that refer to chunk_ids not retrieved
    uncited_sentence_rate: float = float("nan")       # 1 - citation_sentence_rate

    def as_dict(self) -> Dict[str, float]:
        return {
            "n_total": self.n_total,
            "n_with_gold": self.n_with_gold,
            "n_with_answer": self.n_with_answer,
            "recall_at_k": self.recall_at_k,
            "citation_sentence_rate": self.citation_sentence_rate,
            "uncited_sentence_rate": self.uncited_sentence_rate,
            "bad_citation_rate": self.bad_citation_rate,
        }


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_sentences_ja_en(text: str) -> List[str]:
    # Very simple splitter: Japanese punctuation + English punctuation
    # (Good enough for a quick "are sentences cited?" diagnostic)
    parts = re.split(r"(?<=[。．\.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text.strip()]

def split_units(text: str, unit: str) -> list[str]:
    if unit == "line":
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 既存のsentence splitロジックをここに残す
    return split_sentences_ja_en(text)  # 既存関数に合わせて


def eval_recall_at_k(rows: List[dict], retriever, *, k: int) -> Tuple[float, int]:
    hits = 0
    n = 0
    for r in rows:
        gold = r.get("gold_chunk_ids")
        if not gold:
            continue
        n += 1
        res = retriever.search(r["question"], top_k=k, topic=r.get("topic"))
        got = {c.chunk_id for _, c in res}
        if any(g in got for g in gold):
            hits += 1
    return (hits / n) if n else float("nan"), n


def eval_faithfulness(rows: List[dict], retriever, *, k: int, unit: str = "sentence") -> Tuple[float, float, float, int]:
    """
    Very lightweight "faithfulness" proxy:
    - sentence has a citation? (chunk_id, p#)
    - citation refers to a chunk_id that was in retrieved top-k?

    This DOES NOT check semantic entailment. It's meant as a cheap guardrail metric.
    """
    total_units = 0
    cited_units = 0
    bad_cites = 0
    total_cites = 0
    n = 0

    for r in rows:
        ans = r.get("answer")
        if not ans:
            continue
        n += 1

        res = retriever.search(r["question"], top_k=k, topic=r.get("topic"))
        retrieved_ids = {c.chunk_id for _, c in res}

        #sents = split_sentences_ja_en(ans)
        units = split_units(ans, unit)
        for u in units:
            total_units += 1
            cites = CITE_RE.findall(u)
            if cites:
                cited_units += 1
            for (chunk_id, _p) in cites:
                chunk_id = chunk_id.strip()
                total_cites += 1
                if chunk_id not in retrieved_ids:
                    bad_cites += 1

    citation_sentence_rate = (cited_units / total_units) if total_units else float("nan")
    bad_citation_rate = (bad_cites / total_cites) if total_cites else float("nan")
    uncited_sentence_rate = 1.0 - citation_sentence_rate if not np.isnan(citation_sentence_rate) else float("nan")
    return citation_sentence_rate, uncited_sentence_rate, bad_citation_rate, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="index/chunks.jsonl", help="Path to chunks.jsonl")
    ap.add_argument("--evalset", type=str, required=True, help="Path to eval questions jsonl")
    ap.add_argument("--retriever", type=str, default="bm25", choices=["tfidf", "bm25"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--unit", type=str, default="sentence", choices=["sentence", "line"], help="Unit for faithfulness eval")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[0]
    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = (repo / index_path).resolve()

    rows = load_jsonl(Path(args.evalset))

    if args.retriever == "tfidf":
        retr = TfidfRetriever.from_jsonl(index_path)
    else:
        retr = BM25Retriever.from_jsonl(index_path)

    m = Metrics()
    m.n_total = len(rows)

    m.recall_at_k, m.n_with_gold = eval_recall_at_k(rows, retr, k=args.k)
    (m.citation_sentence_rate, m.uncited_sentence_rate, m.bad_citation_rate, m.n_with_answer) = \
    eval_faithfulness(rows, retr, k=args.k, unit=args.unit)

    print(json.dumps(m.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
