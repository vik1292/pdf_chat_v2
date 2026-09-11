#!/bin/env python3
"""ChatPDF Streamlit UI: chat on the left, PDF with per-line source highlights on the right."""
import tempfile
import time
from typing import Dict, List, Optional

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from rag import ChatPDF

st.set_page_config(page_title="ChatPDF", layout="wide")

CITATION_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#9a6324",
]


def color_for(citation_number: int) -> str:
    return CITATION_COLORS[(citation_number - 1) % len(CITATION_COLORS)]


def init_state():
    defaults = {
        "messages": [],
        "assistant": None,
        "docs": {},
        "active_doc": None,
        "highlights": [],
        "scroll_page": None,
        "viewer_nonce": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state["assistant"] is None:
        st.session_state["assistant"] = ChatPDF()


def set_highlights(sources: List[Dict], only_citation: Optional[int] = None):
    """Build viewer highlights from sources.

    Evidence sentences get a solid outline; the rest of the chunk gets a dashed one.
    Only cited sources are shown unless ``only_citation`` isolates a single one.
    """
    chosen = [s for s in sources if s["cited"]] or sources
    if only_citation is not None:
        chosen = [s for s in sources if s["citation_number"] == only_citation]
    highlights = []
    for s in chosen:
        color = color_for(s["citation_number"])
        for box in s["chunk_boxes"]:
            highlights.append(
                {"doc_id": s["doc_id"], "page": s["page"], "box": box, "color": color, "border": "dashed"}
            )
        for box in s["evidence_boxes"]:
            highlights.append(
                {"doc_id": s["doc_id"], "page": s["page"], "box": box, "color": color, "border": "solid"}
            )
    st.session_state["highlights"] = highlights
    if chosen:
        st.session_state["active_doc"] = chosen[0]["doc_id"]
        st.session_state["scroll_page"] = chosen[0]["page"] + 1
    st.session_state["viewer_nonce"] += 1


def on_upload():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["docs"] = {}
    st.session_state["highlights"] = []
    st.session_state["active_doc"] = None
    st.session_state["scroll_page"] = None
    for file in st.session_state["file_uploader"] or []:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file.getbuffer())
            path = tf.name
        with st.spinner(f"Ingesting {file.name}"):
            t0 = time.time()
            doc_id = st.session_state["assistant"].ingest(path, name=file.name)
        st.session_state["docs"][doc_id] = {"name": file.name, "bytes": file.getvalue()}
        st.session_state["active_doc"] = st.session_state["active_doc"] or doc_id
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": f"Ingested {file.name} in {time.time() - t0:.1f}s",
                "sources": [],
            }
        )


def render_sources(sources: List[Dict], msg_idx: int):
    if not sources:
        return
    cited_count = sum(1 for s in sources if s["cited"])
    with st.expander(f"Sources ({cited_count} cited of {len(sources)} retrieved)"):
        for s in sources:
            n = s["citation_number"]
            badge = "cited" if s["cited"] else "not cited"
            st.markdown(
                f'<span style="color:{color_for(n)};font-weight:700">[{n}]</span> '
                f'**{s["doc_name"]}**, page {s["page_label"]} · {badge} · score {s["score"]:.2f}',
                unsafe_allow_html=True,
            )
            if s["evidence_text"]:
                st.markdown(f"> {s['evidence_text']}")
            with st.popover("Full chunk"):
                st.write(s["text"])
            st.button(
                "Show in PDF",
                key=f"show_{msg_idx}_{n}",
                on_click=set_highlights,
                args=(sources, n),
            )
            st.divider()


def render_chat():
    for i, m in enumerate(st.session_state["messages"]):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("sources", []), i)


def handle_question(text: str):
    st.session_state["messages"].append({"role": "user", "content": text, "sources": []})
    with st.spinner("Thinking"):
        response = st.session_state["assistant"].ask(text)
    st.session_state["messages"].append(
        {"role": "assistant", "content": response["answer"], "sources": response["sources"]}
    )
    set_highlights(response["sources"])


def render_pdf():
    docs = st.session_state["docs"]
    if not docs:
        st.info("Upload a PDF to view it here")
        return
    ids = list(docs)
    active = st.session_state["active_doc"] if st.session_state["active_doc"] in docs else ids[0]
    if len(ids) > 1:
        active = st.selectbox(
            "Document", ids, index=ids.index(active), format_func=lambda d: docs[d]["name"]
        )
        if active != st.session_state["active_doc"]:
            st.session_state["active_doc"] = active
            st.session_state["scroll_page"] = None

    annotations = []
    for h in st.session_state["highlights"]:
        if h["doc_id"] != active:
            continue
        x0, top, x1, bottom = h["box"]
        annotations.append(
            {
                "page": h["page"] + 1,  # viewer pages are 1-based
                "x": x0,
                "y": top,
                "width": x1 - x0,
                "height": bottom - top,
                "color": h["color"],
                "border": h["border"],
            }
        )

    pdf_viewer(
        input=docs[active]["bytes"],
        width="100%",
        height=900,
        zoom_level="auto",
        annotations=annotations,
        annotation_outline_size=2,
        render_text=True,
        scroll_to_page=st.session_state["scroll_page"],
        key=f"viewer_{active}_{st.session_state['viewer_nonce']}",
    )


def page():
    init_state()
    st.header("ChatPDF")
    st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        key="file_uploader",
        on_change=on_upload,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.session_state["docs"]:
        left, right = st.columns([1, 1])
        with left:
            render_chat()
        with right:
            st.subheader("PDF")
            render_pdf()
    else:
        render_chat()

    if question := st.chat_input("Ask a question about the documents"):
        if question.strip():
            handle_question(question.strip())
            st.rerun()


if __name__ == "__main__":
    page()
