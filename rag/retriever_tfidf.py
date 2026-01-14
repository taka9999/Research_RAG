from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Lightweight query expansion for econometrics + RL ---
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
    """
    Add a few high-signal synonyms/abbreviations based on keywords in the query.
    Keeps it lightweight; avoids exploding prompt length.
    """
    ql = q.lower()
    extra = []
    for k, vals in EXPAND.items():
        if k in ql:
            extra.extend(vals)

    # Deduplicate while preserving order
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

    def search(self, query: str, *, top_k: int = 8, topic: str | None = None, debug: bool = False) -> List[Tuple[float, Chunk]]:
        
        q_exp = expand_query(query)
        if debug and q_exp != query:
            print("[expanded query]", q_exp)

        qv = self.vectorizer.transform([q_exp])
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

    tests2 = [
        ("proximal policy optimization clip objective KL penalty","reinforcement_learning"),
        ("PPO clipped surrogate objective r_t(theta) advantage","reinforcement_learning"),
        ("trust region KL constraint TRPO PPO","reinforcement_learning"),
        ("KL penalty beta PPO","reinforcement_learning"),
    ]

    for q, t in tests2:
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