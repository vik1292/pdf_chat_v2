from rag import ChatPDF, build_prompt, parse_citations
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker


def make(answer="Up to 5,000 dollars per year [1]."):
    llm = FakeLLM(answer)
    return ChatPDF(llm=llm, embedder=FakeEmbedder(), reranker=FakeReranker()), llm


def test_parse_citations():
    assert parse_citations("See [1] and [3], not [x]. Also [2][2].") == {1, 2, 3}
    assert parse_citations("no citations") == set()


def test_build_prompt_contains_question_and_labeled_blocks():
    prompt = build_prompt("What is covered?", [(1, "a.pdf", 2, "Some context.")])
    assert "Question: What is covered?" in prompt
    assert "[1] (a.pdf, page 2)" in prompt
    assert "Some context." in prompt


def test_ask_without_documents():
    bot, _ = make()
    out = bot.ask("anything")
    assert out["sources"] == []
    assert "upload" in out["answer"].lower()


def test_ingest_registers_document_and_chunks(sample_pdf):
    bot, _ = make()
    doc_id = bot.ingest(sample_pdf)
    rec = bot.documents[doc_id]
    assert rec.name == "benefits.pdf"
    assert len(rec.pages) == 2
    assert rec.chunks and all(c.doc_id == doc_id for c in rec.chunks)
    assert {c.page for c in rec.chunks} == {0, 1}


def test_ask_returns_cited_source_with_evidence_on_correct_page(sample_pdf):
    bot, llm = make()
    bot.ingest(sample_pdf)
    out = bot.ask("How much tuition reimbursement can I get per year?")

    assert out["answer"].startswith("Up to 5,000")
    assert out["sources"], "expected at least one source"
    top = out["sources"][0]
    assert top["citation_number"] == 1
    assert top["cited"] is True
    assert top["page"] == 0 and top["page_label"] == "1"
    assert top["doc_name"] == "benefits.pdf"
    assert "tuition" in top["text"].lower()
    assert "5,000" in top["evidence_text"]
    assert top["evidence_boxes"], "evidence should have coordinates"
    assert top["chunk_boxes"], "chunk should have coordinates"
    page = bot.documents[top["doc_id"]].pages[0]
    for x0, t, x1, b in top["evidence_boxes"]:
        assert 0 <= x0 < x1 <= page.width and 0 <= t < b <= page.height
    assert len(top["evidence_boxes"]) < len(top["chunk_boxes"]), "evidence must be narrower than the chunk"

    # The LLM saw the original question and labeled context, not an expanded query.
    assert "Question: How much tuition reimbursement can I get per year?" in llm.prompts[-1]
    assert "[1] (benefits.pdf, page 1)" in llm.prompts[-1]


def test_uncited_sources_are_flagged_and_all_cited_when_answer_has_no_markers(sample_pdf):
    bot, _ = make(answer="Fifteen days [2].")
    bot.ingest(sample_pdf)
    out = bot.ask("How many days of paid time off do employees accrue?")
    cited = [s["citation_number"] for s in out["sources"] if s["cited"]]
    assert cited == [2]

    bot2, _ = make(answer="No markers here.")
    bot2.ingest(sample_pdf)
    out2 = bot2.ask("paid time off")
    assert out2["sources"] and all(s["cited"] for s in out2["sources"])


def test_multiple_documents_are_searchable_and_sources_carry_doc_identity(two_pdfs):
    bot, _ = make(answer="Hard hats [1].")
    a, b = two_pdfs
    bot.ingest(a)
    bot.ingest(b)
    out = bot.ask("Are hard hats required on the factory floor?")
    assert out["sources"][0]["doc_name"] == "safety.pdf"
    assert out["sources"][0]["doc_id"] in bot.documents


def test_clear_removes_everything(sample_pdf):
    bot, _ = make()
    bot.ingest(sample_pdf)
    bot.clear()
    assert bot.documents == {}
    assert bot.ask("x")["sources"] == []
