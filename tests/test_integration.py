import os

import pytest

from rag import ChatPDF
from retrieval import FastEmbedReranker, FastEmbedder
from tests.fakes import FakeLLM

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.environ.get("PDFCHAT_INTEGRATION") != "1", reason="set PDFCHAT_INTEGRATION=1")
def test_semantic_gap_query_finds_tuition_chunk(sample_pdf):
    bot = ChatPDF(
        llm=FakeLLM("Tuition reimbursement [1]."),
        embedder=FastEmbedder(),
        reranker=FastEmbedReranker(),
    )
    bot.ingest(sample_pdf)
    out = bot.ask("Are there any benefits that help me go back to school or get certifications?")
    top = out["sources"][0]
    assert "tuition" in top["text"].lower()
    assert top["page"] == 0
    assert top["evidence_boxes"]
    evidence = top["evidence_text"].lower()
    assert "reimbursement" in evidence or "certifications" in evidence
