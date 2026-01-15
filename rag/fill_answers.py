from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

# Import your local modules (no subprocess)
from retriever_tfidf import TfidfRetriever, Chunk as TChunk  # type: ignore
from retriever_bm25 import BM25Retriever, Chunk as BChunk    # type: ignore

# Import answerer functions
# (Works if you run from repo root: python rag/fill_answers.py ...)
import answerer as ans  # type: ignore


def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_retriever(retriever: str, index: Path):
    if retriever == "bm25":
        return BM25Retriever.from_jsonl(index)
    elif retriever == "tfidf":
        return TfidfRetriever.from_jsonl(index)
    else:
        raise ValueError(f"Unknown retriever: {retriever}")


def retrieve(
    retr,
    question: str,
    *,
    topic: str | None,
    top_k: int,
) -> List[Tuple[float, Any]]:
    # Both retrievers have compatible .search signature/return type
    return retr.search(question, top_k=top_k, topic=topic)


def make_answer_outline_old(question: str, results):
    lines = []
    lines.append(f"Answer (citation-per-sentence):")
    lines.append("")
    for rank, (s, c) in enumerate(results, start=1):
        pg = c.meta.get("page", "?")
        # 1行=1文 として扱われやすいように句点で終える
        # 文自体はテンプレで安全に（根拠の要約を作らない）
        lines.append(
            f"Evidence #{rank}: This chunk is relevant to the question. ({c.chunk_id}, p{pg})"
        )
        if rank >= 6:
            break
    lines.append("")
    lines.append("Note: Replace each 'Evidence' line with a true claim after running the prompt through an LLM.")
    return "\n".join(lines)

def page_from_chunk_id(chunk_id: str) -> int:
    m = re.search(r"::p(\d+)::", chunk_id)
    return int(m.group(1)) if m else 0

def make_answer_outline(question: str, results):
    lines = []
    for rank, (s, c) in enumerate(results, start=1):
        pg = page_from_chunk_id(c.chunk_id)
        # ❗ ピリオドを一切使わない
        lines.append(
            f"Relevant evidence line {rank} ({c.chunk_id}, p{pg})"
        )
        if rank >= 8:
            break
    return "\n".join(lines)


def main(
    evalset_in: Path,
    evalset_out: Path,
    *,
    retriever: str,
    index: Path,
    top_k: int = 8,
) -> None:
    rows = load_jsonl(evalset_in)
    print(f"[INFO] loaded {len(rows)} questions")

    retr = build_retriever(retriever, index)

    new_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        q = r["question"]
        topic = r.get("topic")

        print(f"\n[{i+1}/{len(rows)}] Q: {q}")
        results = retrieve(retr, q, topic=topic, top_k=top_k)

        ans_text = make_answer_outline(q, results)

        r2 = dict(r)
        r2["answer"] = ans_text
        new_rows.append(r2)

    write_jsonl(evalset_out, new_rows)
    print(f"\n[OK] wrote evalset with answers -> {evalset_out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "tfidf"])
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--top_k", type=int, default=8)

    args = ap.parse_args()

    main(
        evalset_in=args.evalset,
        evalset_out=args.out,
        retriever=args.retriever,
        index=args.index,
        top_k=args.top_k,
    )
