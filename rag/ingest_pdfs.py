# rag/ingest_pdfs.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF


# -------------------------
# Utils
# -------------------------
_WS_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")  # non-breaking space
    s = _WS_RE.sub(" ", s).strip()
    return s

def guess_topic_from_path(pdf_path: Path) -> str:
    """
    Expect: .../data/papers/<topic>/<file>.pdf
    Returns <topic> or "misc".
    """
    parts = list(pdf_path.parts)
    if "papers" in parts:
        i = parts.index("papers")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "misc"

def iter_target_pdfs(base_dir: Path, topics: List[str]) -> Iterable[Path]:
    """
    base_dir: research_rag/data/papers
    topics: e.g. ["econometrics", "reinforcement_learning"]
    """
    for t in topics:
        p = base_dir / t
        if not p.exists():
            continue
        yield from sorted(p.glob("*.pdf"))


# -------------------------
# Paragraph chunking
# -------------------------
def split_into_paragraphs(page_text: str, *, min_chars: int = 80) -> List[str]:
    """
    Very simple paragraph splitter:
    - split by blank lines
    - keep sufficiently long fragments
    """
    raw = [clean_text(x) for x in page_text.split("\n\n")]
    out = [x for x in raw if len(x) >= min_chars]
    return out

def pack_paragraphs(
    paras: List[str],
    *,
    target_chars: int = 1800,
    overlap_paras: int = 1
) -> List[str]:
    """
    Merge consecutive paragraphs into chunks of roughly target_chars.
    Overlap by a few paragraphs to avoid cutting important context.
    """
    if not paras:
        return []
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append(" ".join(cur).strip())
            cur = []
            cur_len = 0

    for p in paras:
        if cur_len + len(p) + 1 <= target_chars:
            cur.append(p)
            cur_len += len(p) + 1
        else:
            flush()
            cur.append(p)
            cur_len = len(p)

    flush()

    # Apply overlap (paragraph-level approximation)
    if overlap_paras > 0 and len(chunks) >= 2:
        # We can't perfectly overlap at paragraph granularity after packing,
        # so we do a light overlap by appending a tail of previous chunk.
        overlapped: List[str] = []
        prev_tail = ""
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
                prev_tail = c[-400:]  # last ~400 chars as tail
            else:
                overlapped.append((prev_tail + " " + c).strip())
                prev_tail = c[-400:]
        chunks = overlapped

    return chunks


# -------------------------
# Main ingest
# -------------------------
def ingest_pdf(
    pdf_path: Path,
    *,
    source_type: str = "paper",
    topic: Optional[str] = None,
    min_chars_para: int = 80,
    target_chars_chunk: int = 1800
) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}
    title = (meta.get("title") or pdf_path.stem).strip()

    topic = topic or guess_topic_from_path(pdf_path)

    records: List[Dict[str, Any]] = []
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text = page.get_text("text")
        text = clean_text(text)
        if not text:
            continue

        paras = split_into_paragraphs(text, min_chars=min_chars_para)
        chunks = pack_paragraphs(paras, target_chars=target_chars_chunk, overlap_paras=1)

        for j, chunk_text in enumerate(chunks):
            rec = {
                "chunk_id": f"{pdf_path.stem}::p{page_idx+1}::c{j}",
                "text": chunk_text,
                "meta": {
                    "source_type": source_type,
                    "topic": [topic],              # list so later you can add more tags
                    "title": title,
                    "page": page_idx + 1,
                    "path": str(pdf_path.as_posix()),
                },
            }
            records.append(rec)

    doc.close()
    return records


def build_chunks_jsonl(
    *,
    repo_root: Path,
    topics: List[str],
    out_path: Path
) -> Tuple[int, int]:
    papers_dir = repo_root / "data" / "papers"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_recs: List[Dict[str, Any]] = []
    pdfs = list(iter_target_pdfs(papers_dir, topics))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {papers_dir} for topics={topics}")

    for pdf in pdfs:
        t = guess_topic_from_path(pdf)
        recs = ingest_pdf(pdf, topic=t)
        all_recs.extend(recs)

    # write
    n_chunks = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_chunks += 1

    return len(pdfs), n_chunks


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]  # .../research_rag
    out_path = repo_root / "index" / "chunks.jsonl"
    topics = ["econometrics", "reinforcement_learning"]

    n_pdfs, n_chunks = build_chunks_jsonl(repo_root=repo_root, topics=topics, out_path=out_path)
    print(f"Done. PDFs: {n_pdfs}, chunks: {n_chunks}")
    print(f"Wrote: {out_path}")
