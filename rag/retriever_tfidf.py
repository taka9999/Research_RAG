from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

EXPAND = {
        "fixed effects": ["FE", "within estimator", "within transformation"],
        "first differencing": ["FD", "first-difference", "first-difference estimator"],
        "hac": ["newey-west", "heteroskedasticity and autocorrelation consistent"],
        "gmm": ["optimal gmm", "two-step", "weighting matrix"],
        }

def expand_query(q: str) -> str:
    qq = q
    ql = q.lower()
    for k, vs in EXPAND.items():
        if k in ql:
            qq += " " + " ".join(vs)
    return qq


@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]


class TfidfRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self.X = self.vectorizer.fit_transform([c.text for c in chunks])

    @classmethod
    def from_jsonl(cls, jsonl_path: str | Path, *, require_skip_false: bool = True) -> "TfidfRetriever":
        chunks: List[Chunk] = []
        p = Path(jsonl_path)

        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            meta = r.get("meta", {})
            if require_skip_false and meta.get("skip"):
                continue
            chunks.append(Chunk(r["chunk_id"], r["text"], meta))

        return cls(chunks)

    def search(self, query: str, *, top_k: int = 8, topic: str | None = None) -> List[Tuple[float, Chunk]]:
        query = expand_query(query)

        qv = self.vectorizer.transform([query])
        scores = (self.X @ qv.T).toarray().reshape(-1)

        idx = np.argsort(-scores)

        out: List[Tuple[float, Chunk]] = []
        for i in idx:
            if scores[i] <= 0:
                break
            c = self.chunks[int(i)]
            if topic is not None:
                if topic not in (c.meta.get("topic") or []):
                    continue
            out.append((float(scores[i]), c))
            if len(out) >= top_k:
                break
        return out


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    retr = TfidfRetriever.from_jsonl(repo / "index" / "chunks.jsonl")

    tests = [
        ("strict exogeneity", "econometrics"),
        ("fixed effects versus first differencing", "econometrics"),
        ("GMM two-step M-estimators HAC", "econometrics"),
        ("policy optimization continuous time PPO", "reinforcement_learning"),
        ("multi-headed networks continual learning", None),
    ]

    for q, t in tests:
        print("\n" + "="*80)
        print("Q:", q, "| topic:", t)
        res = retr.search(q, top_k=5, topic=t)
        for s, c in res:
            print(
                f"- {s:.4f} {c.chunk_id} | "
                f"title={c.meta.get('title','')} | "
                f"topic={c.meta.get('topic',[])} | "
                f"page={c.meta.get('page','')}"
            )
            print("  ", c.text[:160].replace("\n"," "), "...")