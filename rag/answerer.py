# rag/answerer.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import textwrap

from retriever_tfidf import TfidfRetriever, Chunk  # 同じフォルダ構成の場合


def format_context(results: List[Tuple[float, Chunk]], *, max_chars_per_chunk: int = 1400) -> str:
    """
    LLMに渡す（または人間が読む）ための context 整形。
    各chunkに [chunk_id | pX | title | topic | score] を付ける。
    """
    blocks = []
    for score, c in results:
        page = c.meta.get("page", "?")
        title = c.meta.get("title", "")
        topic = c.meta.get("topic", [])
        header = f"[{c.chunk_id} | p{page} | {title} | topic={topic} | score={score:.4f}]"
        body = c.text.strip().replace("\n", " ")
        body = body[:max_chars_per_chunk]
        blocks.append(header + "\n" + body)
    return "\n\n---\n\n".join(blocks)


def draft_answer_template(question: str, results: List[Tuple[float, Chunk]]) -> str:
    """
    LLMなしで、根拠chunkから“回答の骨格”を自動生成する（人間/LLMが埋める用）
    """
    top = results[:6]

    def cite(c: Chunk) -> str:
        return f"({c.chunk_id}, p{c.meta.get('page', '?')})"

    bullets = []
    for score, c in top:
        snippet = c.text.strip().replace("\n", " ")
        snippet = snippet[:260]
        bullets.append(f"- {snippet} {cite(c)}")

    template = f"""\
Question:
{question}

Answer outline (evidence-first):
1) Definition / condition:
   - (Define it precisely; mention conditioning set and timing.)

2) Main explanation:
   - (Key mechanisms / why it matters / intuition)
   - (If relevant: consistency vs efficiency, assumptions, etc.)

3) Practical diagnostics / fixes:
   - (What to test / what to add / how to interpret)

Evidence bullets (auto-extracted):
{chr(10).join(bullets)}

Notes / follow-ups:
- (If evidence is insufficient, say what's missing and what to retrieve.)
"""
    return textwrap.dedent(template)


# --- Prompt templates (copy-paste) ---
PROMPT_EXPLORE = """\
あなたは研究アシスタントです。以下の CONTEXT（抜粋）を優先して使い、日本語で分かりやすく答えてください。

【ルール（探索モード）】
- CONTEXTを最優先で使う。
- CONTEXT外の一般知識で補足してよいが、その場合は「（一般知識として補足）」と明示する。
- 断定しすぎず、前提・限界・代替解釈を短く添える。
- 重要な事実主張（定義・定理・数式・主張）には、可能な限り引用を付ける： (chunk_id, p#)
- CONTEXTが不足している場合は「不足」と言い、追加で検索すべきクエリ案を3つ提示する。

【出力フォーマット】
1) 要点（3〜6行）
2) 丁寧な説明（必要なら数式/直観/例）
3) 追加で確認したい点 / 追加retrievalクエリ案（3つ）

質問:
{question}

CONTEXT:
{context}
"""

PROMPT_STRICT = """\
あなたは厳密な研究アシスタントです。以下の CONTEXT（抜粋）のみを根拠に、日本語で回答してください。

【ルール（厳密モード）】
- CONTEXT以外の知識は一切使わない（推測・一般常識の補足も禁止）。
- すべての重要な主張（各箇条書き、または各文）に必ず引用を付ける： (chunk_id, p#)
- CONTEXTから直接言えない場合は、必ず「不足」と明示し、何が不足かを書き、追加で必要な検索クエリ案を提示する。
- 可能なら、前提条件（Assumption等）と、その含意（consistency/efficiency等）を分けて整理する。

【出力フォーマット】
A) 結論（引用付き）
B) 根拠（箇条書き：主張 + 引用）
C) 不足している情報（あれば）
D) 追加retrievalクエリ案（3つ）

質問:
{question}

CONTEXT:
{context}
"""


def build_chat_prompt(question: str, context: str, mode: str = "explore") -> str:
    mode = (mode or "explore").lower()
    if mode not in ("explore", "strict"):
        raise ValueError(f"Unknown mode={mode}. Use 'explore' or 'strict'.")
    tmpl = PROMPT_EXPLORE if mode == "explore" else PROMPT_STRICT
    return tmpl.format(question=question.strip(), context=context.strip())


def run_one_question(
    retriever: TfidfRetriever,
    question: str,
    topic: Optional[str],
    *,
    mode: str = "explore",
    top_k: int = 8,
    print_outline: bool = True,
    max_chars_per_chunk: int = 1400,
) -> None:
    results = retriever.search(question, top_k=top_k, topic=topic)

    context = format_context(results, max_chars_per_chunk=max_chars_per_chunk)
    prompt = build_chat_prompt(question, context, mode=mode)

    print("=" * 88)
    print(f"Question: {question}")
    print(f"Mode: {mode} | topic: {topic} | top_k: {top_k}")

    print("\nEvidence (top chunks):")
    for s, c in results:
        print(f"- {s:.4f} {c.chunk_id} (p{c.meta.get('page','?')}, title={c.meta.get('title','')})")

    if print_outline:
        print("\n" + "=" * 88)
        print("DRAFT OUTLINE (no-LLM):\n")
        print(draft_answer_template(question, results))

    print("\n" + "=" * 88)
    print("PASTE INTO CHATGPT:\n")
    print(prompt)
    print("\n" + "=" * 88)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="explore", choices=["explore", "strict"])
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_chars", type=int, default=1400)
    parser.add_argument("--no_outline", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    retr = TfidfRetriever.from_jsonl(repo / "index" / "chunks.jsonl")

    QUESTIONS = [
        {"question": "Why does strict exogeneity matter for FE?", "topic": "econometrics"},
        {"question": "Explain PPO objective with KL.", "topic": "reinforcement_learning"},
    ]

    for q in QUESTIONS:
        run_one_question(
            retriever=retr,
            question=q["question"],
            topic=q.get("topic"),
            mode=args.mode,
            top_k=args.top_k,
            print_outline=not args.no_outline,
            max_chars_per_chunk=args.max_chars,
        )


if __name__ == "__main__":
    main()
