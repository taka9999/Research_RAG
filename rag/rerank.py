from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the Chunk type from your existing retriever module.
# This keeps the reranker decoupled from which retriever you use (TF-IDF/BM25/etc).
from retriever_tfidf import Chunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: List[Tuple[float, Chunk]], *, top_k: int) -> List[Tuple[float, Chunk]]:
        ...


@dataclass
class IdentityReranker:
    """No-op reranker (keeps original ranking)."""
    def rerank(self, query: str, candidates: List[Tuple[float, Chunk]], *, top_k: int) -> List[Tuple[float, Chunk]]:
        return candidates[:top_k]


@dataclass
class MMRReranker:
    """
    Maximal Marginal Relevance (MMR) reranker.
    - Improves diversity in retrieved chunks (reduces near-duplicates),
      while keeping relevance.

    This is a "minimal" reranker: it does not require external models,
    and runs fast on small candidate pools.

    lambda_ in [0,1]:
      - closer to 1 => prioritize relevance
      - closer to 0 => prioritize diversity
    """
    lambda_: float = 0.7
    stop_words: str | None = "english"
    ngram_range: tuple[int, int] = (1, 2)

    def rerank(self, query: str, candidates: List[Tuple[float, Chunk]], *, top_k: int) -> List[Tuple[float, Chunk]]:
        if not candidates:
            return []
        top_k = min(top_k, len(candidates))

        texts = [c.text for _, c in candidates]
        vect = TfidfVectorizer(lowercase=True, stop_words=self.stop_words, ngram_range=self.ngram_range)
        X = vect.fit_transform([query] + texts)
        qv = X[0]
        dv = X[1:]

        # cosine similarities to query
        rel = (dv @ qv.T).toarray().reshape(-1)

        # cosine similarity between docs
        doc_sim = (dv @ dv.T).toarray()
        # normalize diagonal to 0 to avoid self selection issues
        np.fill_diagonal(doc_sim, 0.0)

        selected: List[int] = []
        remaining = set(range(len(candidates)))

        # pick the most relevant first
        first = int(np.argmax(rel))
        selected.append(first)
        remaining.remove(first)

        while len(selected) < top_k and remaining:
            best_i = None
            best_score = -1e18
            for i in list(remaining):
                max_sim_to_selected = max(doc_sim[i, j] for j in selected) if selected else 0.0
                mmr = self.lambda_ * rel[i] - (1.0 - self.lambda_) * max_sim_to_selected
                if mmr > best_score:
                    best_score = float(mmr)
                    best_i = i
            assert best_i is not None
            selected.append(best_i)
            remaining.remove(best_i)

        # Return in MMR selection order, but keep original scores (and optionally add mmr score if you want)
        return [candidates[i] for i in selected]
