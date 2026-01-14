# rag/answerer_api.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI

# 既存のretrieverを再利用
from rag.retriever_tfidf import TfidfRetriever, Chunk


def format_context(results: List[Tuple[float, Chunk]], *, max_chars_per_chunk: int = 1200) -> str:
    blocks = []
    for score, c in results:
        header = (
            f"[{c.chunk_id} | p{c.meta.get('page')} | "
            f"{c.meta.get('title','')} | topic={c.meta.get('topic',[])} | score={score:.4f}]"
        )
        body = c.text.strip().replace("\n", " ")
        body = body[:max_chars_per_chunk]
        blocks.append(header + "\n" + body)
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    # 研究用途：引用強制・根拠外推測禁止
    return f"""\
# Task
Answer the QUESTION in Japanese using ONLY the CONTEXT.

# Rules (strict)
- Every factual claim must have at least one citation in the form: (chunk_id, p#).
- If the context is insufficient to answer part of the question, say what is missing and suggest what to retrieve next.
- Do not invent citations. Do not cite anything not in CONTEXT.

# QUESTION
{question}

# CONTEXT
{context}

# Output format
- Start with a short structured answer (3-6 bullets).
- Then add a brief explanation section.
- End with a "Citations used" list (chunk_id, page) (deduplicated).
"""


def call_openai_responses(prompt: str, *, model: str, temperature: float) -> str:
    client = OpenAI()
    # Responses API（推奨） :contentReference[oaicite:3]{index=3}
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )
    return resp.output_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--topic", type=str, default=None, help="e.g., econometrics / reinforcement_learning")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--model", type=str, default="gpt-5.2")  # 例：quickstartに合わせたデフォルト :contentReference[oaicite:4]{index=4}
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    retr = TfidfRetriever.from_jsonl(repo / "index" / "chunks.jsonl")

    results = retr.search(args.question, top_k=args.top_k, topic=args.topic)
    if not results:
        print("No retrieval results. Try changing the query or topic.")
        return

    context = format_context(results, max_chars_per_chunk=1200)
    prompt = build_prompt(args.question, context)

    answer = call_openai_responses(prompt, model=args.model, temperature=args.temperature)
    print(answer)


if __name__ == "__main__":
    main()
