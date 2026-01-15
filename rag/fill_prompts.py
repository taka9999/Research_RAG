from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import answerer  # rag/answerer.py（同ディレクトリなのでこれでimportできる）


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


def main(
    evalset: Path,
    out: Path,
    *,
    retriever: str,
    index: Path,
    mode: str,
    top_k: int,
    pool_k: int,
    rerank: str,
    max_chars: int,
) -> None:
    rows = load_jsonl(evalset)
    print(f"[INFO] loaded {len(rows)} questions")

    new_rows = []
    for i, r in enumerate(rows, start=1):
        q = r["question"]
        topic = r.get("topic")
        print(f"[{i}/{len(rows)}] making prompt: {q}")

        prompt = answerer.make_prompt(
            question=q,
            topic=topic,
            retriever_name=retriever,
            index_path=index,
            rerank=rerank,
            top_k=top_k,
            pool_k=pool_k,
            max_chars_per_chunk=max_chars,
            mode=mode,
        )

        r2 = dict(r)
        r2["prompt"] = prompt
        r2["prompt_meta"] = {
            "retriever": retriever,
            "rerank": rerank,
            "top_k": top_k,
            "pool_k": pool_k,
            "max_chars": max_chars,
            "mode": mode,
        }
        new_rows.append(r2)

    write_jsonl(out, new_rows)
    print(f"[OK] wrote prompts -> {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "tfidf"])
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--mode", default="cite_strict", choices=["cite_strict", "explore"])
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--pool_k", type=int, default=30)
    ap.add_argument("--rerank", default="none", choices=["none", "mmr"])
    ap.add_argument("--max_chars", type=int, default=1400)

    args = ap.parse_args()

    main(
        evalset=args.evalset,
        out=args.out,
        retriever=args.retriever,
        index=args.index,
        mode=args.mode,
        top_k=args.top_k,
        pool_k=args.pool_k,
        rerank=args.rerank,
        max_chars=args.max_chars,
    )
