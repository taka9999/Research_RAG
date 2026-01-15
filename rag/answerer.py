# rag/answerer.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Protocol, Any
import argparse
import textwrap

from retriever_tfidf import TfidfRetriever, Chunk
from retriever_bm25 import BM25Retriever
from rerank import IdentityReranker, MMRReranker, Reranker


# -------------------------
# Context formatting
# -------------------------
def format_context(
    results: List[Tuple[float, Chunk]],
    *,
    max_chars_per_chunk: int = 1400,
    evidence_k: int = 3,
    tail_frac: float = 0.35,
    weight_mode: str = "ratio",  # "ratio" or "softmax"
    softmax_tau: float = 1.0,
) -> str:
    """
    LLMに渡す（または人間が読む）ための context 整形。

    - 先頭 evidence_k 件を Evidence として優先（長めに入れる）
    - 残りは Support として短めに入れる（tail_frac 倍）
    - スコアはクエリごとにスケールが変わるため、rank と weight(0-1) を付与

    header 例:
      [#1 | w=0.62 | score=9.31 | chunk_id | p12 | title | topic=...]
    """
    if not results:
        return ""

    scores = [float(s) for s, _ in results]
    max_s = max(scores) if scores else 0.0

    # Optional: softmax weights for nicer distribution
    if weight_mode == "softmax":
        import math
        tau = float(softmax_tau) if softmax_tau > 0 else 1.0
        exps = [math.exp((s - max_s) / tau) for s in scores]
        Z = sum(exps) if exps else 1.0
        weights = [e / Z for e in exps]
    else:
        denom = max_s if max_s > 0 else 1.0
        weights = [s / denom for s in scores]

    blocks: List[str] = []
    k_ev = max(0, int(evidence_k))
    tail_chars = max(100, int(max_chars_per_chunk * float(tail_frac)))

    def one_block(rank: int, score: float, w: float, c: Chunk, *, cap: int) -> str:
        page = c.meta.get("page", "?")
        title = c.meta.get("title", "")
        topic = c.meta.get("topic", [])
        header = (
            f"[#{rank} | w={w:.2f} | score={score:.4f} | {c.chunk_id} | "
            f"p{page} | {title} | topic={topic}]"
        )
        body = c.text.strip().replace("\n", " ")
        body = body[:cap]
        return header + "\n" + body

    if k_ev > 0:
        blocks.append("## Evidence (high priority)")

    for i, (score, c) in enumerate(results, start=1):
        cap = max_chars_per_chunk if (i <= k_ev) else tail_chars
        blocks.append(one_block(i, float(score), float(weights[i - 1]), c, cap=cap))

    if k_ev > 0 and len(results) > k_ev:
        # Insert a section divider between evidence and support for readability
        insert_at = 1 + k_ev  # after Evidence header + evidence blocks
        blocks.insert(insert_at, "## Support (lower priority)")

    return "\n\n---\n\n".join(blocks)


def draft_answer_template(question: str, results: List[Tuple[float, Chunk]]) -> str:
    """
    Without using an LLM, create an evidence-first answer outline
    that a human/LLM can fill in.
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
   - (Define it precisely; mention conditioning set and timing.) (chunk_id, p#)

2) Main explanation:
   - (Key mechanisms / why it matters / intuition) (chunk_id, p#)
   - (If relevant: consistency vs efficiency, assumptions, etc.) (chunk_id, p#)

3) Practical diagnostics / fixes:
   - (What to test / what to add / how to interpret) (chunk_id, p#)

Evidence bullets (auto-extracted):
{chr(10).join(bullets)}

Notes / follow-ups:
- (If evidence is insufficient, say what's missing and what to retrieve.)
"""
    return textwrap.dedent(template)


# -------------------------
# Prompt templates
# -------------------------
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

# A stricter template that *forces* citations per sentence/claim.
PROMPT_CITE_STRICT = """\
あなたは厳密な研究アシスタントです。以下の CONTEXT（抜粋）のみを根拠に、日本語で回答してください。

【ルール（引用強制モード）】
- CONTEXT以外の知識は一切使わない（推測・一般常識の補足も禁止）。
- 重要な主張は「1文=1主張」を意識し、各文の末尾に必ず引用を付ける： (chunk_id, p#)
- 1つの文で複数の主張をしない（分割する）。
- CONTEXTから直接言えない場合は、必ず「不足」と明示し、何が不足かを書き、追加で必要な検索クエリ案を3つ提示する。

【出力フォーマット】
A) 結論（各文に引用）
B) 根拠（箇条書き：各行=1主張 + 引用）
C) 不足している情報（あれば）
D) 追加retrievalクエリ案（3つ）

質問:
{question}

CONTEXT:
{context}
"""


def build_chat_prompt(question: str, context: str, mode: str = "explore") -> str:
    mode = (mode or "explore").lower()
    if mode not in ("explore", "cite_strict"):
        raise ValueError(f"Unknown mode={mode}. Use 'explore' or 'cite_strict'.")
    tmpl = PROMPT_EXPLORE if mode == "explore" else PROMPT_CITE_STRICT
    return tmpl.format(question=question.strip(), context=context.strip())

def make_prompt(
    *,
    question: str,
    topic: Optional[str] = None,
    retriever_name: str = "bm25",
    index_path: Optional[Path] = None,
    rerank: str = "none",
    top_k: int = 8,
    pool_k: int = 30,
    max_chars_per_chunk: int = 1400,
    mode: str = "cite_strict",
) -> str:
    """Build a chat prompt (cite_strict or explore) for a single question."""
    repo = Path(__file__).resolve().parents[1]
    idx = Path(index_path).expanduser() if index_path else (repo / "index" / "chunks.jsonl")
    retr = BM25Retriever.from_jsonl(idx) if retriever_name == "bm25" else TfidfRetriever.from_jsonl(idx)
    reranker: Reranker = MMRReranker() if rerank == "mmr" else IdentityReranker()

    pool = retr.search(question, top_k=pool_k, topic=topic)
    results = reranker.rerank(question, pool, top_k=top_k)
    context = format_context(results, max_chars_per_chunk=max_chars_per_chunk)
    return build_chat_prompt(question, context, mode=mode)


# -------------------------
# Runner
# -------------------------
class RetrieverLike(Protocol):
    def search(self, query: str, *, top_k: int = 8, topic: str | None = None, debug: bool = False) -> List[Tuple[float, Chunk]]:
        ...


def run_one_question(
    retriever: RetrieverLike,
    question: str,
    topic: Optional[str],
    *,
    mode: str = "explore",
    top_k: int = 8,
    pool_k: int = 30,
    reranker: Optional[Reranker] = None,
    print_outline: bool = True,
    max_chars_per_chunk: int = 1400,
) -> None:
    """
    Minimal rerank design:
    - retrieve a larger candidate pool (pool_k)
    - rerank to top_k (Identity/MMR or future LLM reranker)
    """
    pool = retriever.search(question, top_k=pool_k, topic=topic)
    if reranker is None:
        reranker = IdentityReranker()
    results = reranker.rerank(question, pool, top_k=top_k)

    context = format_context(results, max_chars_per_chunk=max_chars_per_chunk)
    prompt = build_chat_prompt(question, context, mode=mode)

    print("=" * 88)
    print(f"Question: {question}")
    print(f"Mode: {mode} | topic: {topic} | top_k: {top_k} | pool_k: {pool_k} | rerank: {type(reranker).__name__}")

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
    parser.add_argument("--mode", type=str, default="explore", choices=["explore", "cite_strict"])
    parser.add_argument("--retriever", type=str, default="bm25", choices=["tfidf", "bm25"])
    parser.add_argument("--rerank", type=str, default="none", choices=["none", "mmr"])
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--pool_k", type=int, default=30)
    parser.add_argument("--max_chars", type=int, default=1400)
    parser.add_argument("--no_outline", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().name == "answerer.py") else Path(__file__).resolve().parents[0]

    index_path = repo / "index" / "chunks.jsonl"

    if args.retriever == "tfidf":
        retr = TfidfRetriever.from_jsonl(index_path)
    else:
        retr = BM25Retriever.from_jsonl(index_path)

    reranker = IdentityReranker() if args.rerank == "none" else MMRReranker(lambda_=0.7)

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
            pool_k=args.pool_k,
            reranker=reranker,
            print_outline=not args.no_outline,
            max_chars_per_chunk=args.max_chars,
        )


if __name__ == "__main__":
    main()
