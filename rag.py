"""ChatPDF: multi-document RAG with hybrid retrieval, reranking and precise source locations."""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pdf_extract import (
    BBox,
    Chunk,
    PageText,
    Word,
    chunk_words,
    extract_pages,
    line_boxes,
    sentence_spans,
    span_text,
)
from retrieval import Embedder, HybridIndex, Reranker, select_evidence

_CITATION_RE = re.compile(r"\[(\d+)\]")

PROMPT_TEMPLATE = """You answer questions about uploaded documents.

Rules:
1. Use ONLY the context blocks below. Never use outside knowledge.
2. Cite every fact with the block number in square brackets, e.g. [1] or [2][3].
3. Quote exact figures, dates and names from the context.
4. If the context does not answer the question, reply exactly: I don't have information about that in the provided documents.

Context:
{context}

Question: {question}

Answer with citations:"""

NO_DOCS_MESSAGE = "Please upload a PDF document first."
NO_INFO_MESSAGE = "I don't have information about that in the provided documents."


@dataclass
class DocumentRecord:
    doc_id: str
    name: str
    path: str
    pages: List[PageText]
    chunks: List[Chunk]


def parse_citations(answer: str) -> Set[int]:
    return {int(m) for m in _CITATION_RE.findall(answer)}


def build_prompt(question: str, blocks: Sequence[Tuple[int, str, int, str]]) -> str:
    context = "\n\n".join(f"[{n}] ({doc}, page {page})\n{text}" for n, doc, page, text in blocks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


class ChatPDF:
    def __init__(
        self,
        llm=None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        model_name: str = "mistral:7b",
        candidates: int = 20,
        top_n: int = 5,
    ):
        self._llm = llm
        self._model_name = model_name
        self._embedder = embedder
        self._reranker = reranker
        self.candidates = candidates
        self.top_n = top_n
        self.documents: Dict[str, DocumentRecord] = {}
        self._chunks: Dict[str, Chunk] = {}
        self._index: Optional[HybridIndex] = None

    # ---- lazy heavy dependencies -------------------------------------------------
    @property
    def llm(self):
        if self._llm is None:
            from langchain_ollama import ChatOllama

            self._llm = ChatOllama(model=self._model_name, temperature=0)
        return self._llm

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            from retrieval import FastEmbedder

            self._embedder = FastEmbedder()
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            from retrieval import FastEmbedReranker

            self._reranker = FastEmbedReranker()
        return self._reranker

    @property
    def index(self) -> HybridIndex:
        if self._index is None:
            self._index = HybridIndex(self.embedder)
        return self._index

    # ---- ingest --------------------------------------------------------------------
    def ingest(self, pdf_path: str, name: Optional[str] = None) -> str:
        """Extract, chunk and index one PDF. Returns its doc_id."""
        doc_id = uuid.uuid4().hex[:8]
        pages = extract_pages(pdf_path)
        chunks: List[Chunk] = []
        for page in pages:
            chunks.extend(chunk_words(page.words, doc_id=doc_id, page=page.page))
        record = DocumentRecord(
            doc_id=doc_id,
            name=name or os.path.basename(pdf_path),
            path=pdf_path,
            pages=pages,
            chunks=chunks,
        )
        self.documents[doc_id] = record
        for c in chunks:
            self._chunks[c.chunk_id] = c
        self.index.add([c.chunk_id for c in chunks], [c.text for c in chunks])
        return doc_id

    # ---- ask -----------------------------------------------------------------------
    def ask(self, question: str) -> dict:
        """Answer a question. Returns {"answer": str, "sources": [source dict, ...]}."""
        question = question.strip()
        if not self.documents or not question:
            return {"answer": NO_DOCS_MESSAGE, "sources": []}

        fused = self.index.search(question, k=self.candidates)
        chunks = [self._chunks[cid] for cid, _ in fused]
        if not chunks:
            return {"answer": NO_INFO_MESSAGE, "sources": []}

        scores = self.reranker.score(question, [c.text for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda cs: -cs[1])[: self.top_n]

        blocks: List[Tuple[int, str, int, str]] = []
        sources: List[dict] = []
        for n, (chunk, score) in enumerate(ranked, start=1):
            record = self.documents[chunk.doc_id]
            words = record.pages[chunk.page].words
            context = self._neighbor_context(record, chunk)
            ev_spans = self._evidence_spans(question, words, chunk)
            evidence_boxes: List[BBox] = []
            for s, e in ev_spans:
                evidence_boxes.extend(line_boxes(words, s, e))
            blocks.append((n, record.name, chunk.page + 1, context))
            sources.append(
                {
                    "citation_number": n,
                    "doc_id": chunk.doc_id,
                    "doc_name": record.name,
                    "page": chunk.page,
                    "page_label": str(chunk.page + 1),
                    "text": chunk.text,
                    "context": context,
                    "score": float(score),
                    "cited": False,
                    "evidence_text": " ".join(span_text(words, s, e) for s, e in ev_spans),
                    "evidence_boxes": evidence_boxes,
                    "chunk_boxes": line_boxes(words, chunk.start, chunk.end),
                }
            )

        answer = self.llm.invoke(build_prompt(question, blocks)).content.strip()
        cited = parse_citations(answer) & {s["citation_number"] for s in sources}
        for s in sources:
            s["cited"] = (s["citation_number"] in cited) if cited else True
        return {"answer": answer, "sources": sources}

    def _neighbor_context(self, record: DocumentRecord, chunk: Chunk) -> str:
        """The chunk plus its immediate neighbors on the same page, for the LLM."""
        same_page = [c for c in record.chunks if c.page == chunk.page]
        i = same_page.index(chunk)
        window = same_page[max(0, i - 1) : i + 2]
        return " ".join(c.text for c in window)

    def _evidence_spans(self, question: str, words: List[Word], chunk: Chunk) -> List[Tuple[int, int]]:
        """Sentences within the chunk that best answer the question, by cross-encoder score."""
        spans = sentence_spans(words, chunk.start, chunk.end)
        if len(spans) <= 1:
            return spans
        scores = self.reranker.score(question, [span_text(words, s, e) for s, e in spans])
        return [spans[i] for i in select_evidence(scores)]

    # ---- clear ---------------------------------------------------------------------
    def clear(self) -> None:
        if self._index is not None:
            self._index.close()
        self._index = None
        self.documents = {}
        self._chunks = {}
