"""Hybrid retrieval (BM25 + cosine vectors, reciprocal rank fusion) and reranking."""
from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank), rank starting at 1."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: List[str]) -> List[float]: ...


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 via fastembed. ``embed_query`` adds the BGE query instruction."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class FastEmbedReranker:
    """Cross-encoder reranker via fastembed. Scores are logits; higher is more relevant."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, texts)]


class HybridIndex:
    """Cosine vector search (Chroma, in-process) fused with BM25 keyword search."""

    def __init__(self, embedder: Embedder, collection_name: Optional[str] = None):
        self._embedder = embedder
        self._client = chromadb.EphemeralClient()
        self._name = collection_name or f"pdfchat_{uuid.uuid4().hex}"
        self._collection = self._client.create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._bm25: Optional[BM25Okapi] = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: List[str], texts: List[str]) -> None:
        if not ids:
            return
        self._collection.add(
            ids=list(ids), documents=list(texts), embeddings=self._embedder.embed(list(texts))
        )
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._bm25 = BM25Okapi([tokenize(t) or [""] for t in self._texts])

    def search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        if not self._ids:
            return []
        n = min(k, len(self._ids))
        vec = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)], n_results=n
        )
        vector_ranking = list(vec["ids"][0])

        bm25_scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-bm25_scores, kind="stable")[:n]
        keyword_ranking = [self._ids[i] for i in order if bm25_scores[i] > 0]

        return rrf_fuse([vector_ranking, keyword_ranking])[:k]

    def close(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._ids, self._texts, self._bm25 = [], [], None


def select_evidence(scores: Sequence[float], margin: float = 2.0, max_items: int = 3) -> List[int]:
    """Indices of the best-scoring items: the top one plus any within ``margin`` of it, capped."""
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    best = scores[order[0]]
    chosen = [i for i in order if scores[i] >= best - margin][:max_items]
    return sorted(chosen)
