from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Reuse the same lightweight query expansion dictionary/logic as TF-IDF retriever.
# Keep a local copy to avoid tight coupling between modules.
EXPAND = {
    # Panel / FE / FD
    "fixed effects": ["FE", "within estimator", "within transformation", "demeaning"],
    "first differencing": ["FD", "first-difference", "difference estimator"],
    "random effects": ["RE", "GLS", "quasi-demeaning"],
    "strict exogeneity": ["strictly exogenous", "FE.1", "FD.1", "leads test"],
    "hausman": ["hausman test", "endogeneity test", "specification test"],

    # IV / GMM / HAC
    "instrument": ["IV", "2SLS", "GMM"],
    "endogeneity": ["IV", "2SLS", "control function", "hausman"],
    "gmm": ["two-step", "weighting matrix", "moment conditions", "optimal gmm"],
    "hac": ["Newey-West", "long-run variance", "kernel", "bandwidth"],

    # RL / policy gradient / PPO
    "policy gradient": ["REINFORCE", "advantage", "baseline"],
    "ppo": ["clip", "KL", "policy ratio", "surrogate objective"],
    "kl": ["KL divergence", "relative entropy", "trust region"],
    "actor critic": ["advantage", "value function", "TD error"],
    "temporal difference": ["TD", "TD error", "bootstrapping"],
    "continuous time": ["SDE", "HJB", "Ito", "controlled diffusion"],
    "stochastic control": ["HJB", "Bellman", "dynamic programming"],
}

def expand_query(q: str) -> str:
    ql = q.lower()
    extra: List[str] = []
    for k, vals in EXPAND.items():
        if k in ql:
            extra.extend(vals)

    seen = set()
    extra2 = []
    for w in extra:
        wl = w.lower()
        if wl not in seen:
            extra2.append(w)
            seen.add(wl)

    if extra2:
        return q + " " + " ".join(extra2)
    return q


@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]


class BM25Retriever:
    """
    A lightweight BM25 retriever (Okapi BM25) implemented on top of sklearn's CountVectorizer.

    Design goals:
    - Drop-in replacement for TfidfRetriever.search() signature/return type
    - Keep dependencies minimal (no external bm25 package)
    - Support topic filtering consistent with current code
    """

    def __init__(
        self,
        chunks: List[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        lowercase: bool = True,
        stop_words: str | None = "english",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.chunks = chunks
        self.k1 = float(k1)
        self.b = float(b)

        self.vectorizer = CountVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )

        # Document-term matrix (counts)
        self.X = self.vectorizer.fit_transform([c.text for c in chunks]).tocsr()

        # Document lengths and avgdl
        self.dl = np.asarray(self.X.sum(axis=1)).reshape(-1).astype(float)
        self.avgdl = float(self.dl.mean()) if self.dl.size else 0.0

        # IDF with a BM25-style smoothing:
        # idf(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )
        df = np.asarray((self.X > 0).sum(axis=0)).reshape(-1).astype(float)
        N = float(self.X.shape[0])
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    @classmethod
    def from_jsonl(cls, jsonl_path: str | Path, *, require_skip_false: bool = True, **kwargs) -> "BM25Retriever":
        chunks: List[Chunk] = []
        p = Path(jsonl_path)
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            meta = r.get("meta", {})
            if require_skip_false and meta.get("skip"):
                continue
            chunks.append(Chunk(r["chunk_id"], r["text"], meta))
        return cls(chunks, **kwargs)

    def _bm25_scores_from_query_terms(self, q_term_ids: np.ndarray) -> np.ndarray:
        """
        Compute BM25 scores for all documents given query term indices.
        """
        if q_term_ids.size == 0:
            return np.zeros(self.X.shape[0], dtype=float)

        scores = np.zeros(self.X.shape[0], dtype=float)

        # Precompute denominator length normalization per doc
        # denom = tf + k1*(1 - b + b*dl/avgdl)
        # We'll compute it per-term using sparse access.
        K = self.k1 * (1.0 - self.b + self.b * (self.dl / (self.avgdl + 1e-12)))

        # For each term, add contribution using sparse column slice
        # NOTE: X is CSR; column slicing is more efficient in CSC, but for modest corpora this is OK.
        # If corpus grows large, consider storing X as CSC too.
        for t in q_term_ids:
            col = self.X[:, int(t)]
            tf = np.asarray(col.toarray()).reshape(-1)
            if tf.max() <= 0:
                continue
            numer = tf * (self.k1 + 1.0)
            denom = tf + K
            scores += self.idf[int(t)] * (numer / (denom + 1e-12))

        return scores

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        topic: str | None = None,
        debug: bool = False,
    ) -> List[Tuple[float, Chunk]]:
        q_exp = expand_query(query)
        if debug and q_exp != query:
            print("[expanded query]", q_exp)

        # Build query term indices
        q_counts = self.vectorizer.transform([q_exp])
        q_term_ids = q_counts.indices  # unique term indices present in query

        scores = self._bm25_scores_from_query_terms(q_term_ids)
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
    retr = BM25Retriever.from_jsonl(repo / "index" / "chunks.jsonl")

    tests = [
        ("strict exogeneity", "econometrics"),
        ("fixed effects versus first differencing", "econometrics"),
        ("GMM two-step M-estimators HAC", "econometrics"),
        ("policy optimization continuous time PPO", "reinforcement_learning"),
        ("multi-headed networks continual learning", None),
    ]

    for q, t in tests:
        print("\n" + "=" * 80)
        print("Q:", q, "| topic:", t)
        res = retr.search(q, top_k=5, topic=t)
        for s, c in res:
            print(
                f"- {s:.4f} {c.chunk_id} | "
                f"title={c.meta.get('title','')} | "
                f"topic={c.meta.get('topic',[])} | "
                f"page={c.meta.get('page','')}"
            )
            print("  ", c.text[:160].replace("\n", " "), "...")
