"""Deterministic stand-ins for the embedder, reranker and LLM used in unit tests."""
import hashlib
import math
import re
from typing import List

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class FakeEmbedder:
    """Hashed bag-of-words embedding: texts sharing words get high cosine similarity."""

    dim = 256

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


class FakeReranker:
    """Scores a passage by the number of query tokens it contains."""

    def score(self, query: str, texts: List[str]) -> List[float]:
        q = set(_tokens(query))
        return [float(len(q & set(_tokens(t)))) for t in texts]


class FakeLLM:
    """Records the prompt it received and returns a canned answer."""

    def __init__(self, answer: str = "The answer is here [1]."):
        self.answer = answer
        self.prompts: List[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)

        class _Msg:
            content = self.answer

        return _Msg()
