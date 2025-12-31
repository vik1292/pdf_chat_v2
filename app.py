#!/bin/env python3
import os
import time
import tempfile
import streamlit as st
from streamlit_chat import message
from streamlit_pdf_viewer import pdf_viewer
from rag import ChatPDF
from typing import List, Dict

st.set_page_config(page_title="ChatPDF", layout="wide")


def display_sources_expander(sources: List[Dict], message_idx: int):
    """
    Display expandable source references with numbered citations

    Args:
        sources: List of source dictionaries
        message_idx: Index of message (for unique key)
    """
    if not sources:
        return

    with st.expander(f"📄 View {len(sources)} Source(s)", expanded=False):
        for source in sources:
            citation_num = source.get("citation_number", "?")
            page_label = source.get("page_label", "?")
            page_num = source.get("page", 0)
            chunk_text = source.get("text", "")
            score = source.get("score", 0.0)
            coordinates = source.get("coordinates", [])

            st.markdown(f"**[{citation_num}] Page {page_label}** (PDF page index: {page_num}, Relevance: {score:.2f})")
            st.caption(f"Coordinates found: {len(coordinates)}")
            st.text_area(
                label=f"Source text {citation_num}",
                value=chunk_text,
                height=100,
                key=f"source_{message_idx}_{citation_num}",
                disabled=True
            )
            st.divider()


def display_messages():
    st.subheader("Chat")
    for i, msg_data in enumerate(st.session_state["messages"]):
        # Handle both old format (tuple) and new format (dict)
        if isinstance(msg_data, tuple):
            msg, is_user = msg_data
            message(msg, is_user=is_user, key=str(i))
        else:
            # New format with sources
            msg = msg_data.get("content", "")
            is_user = msg_data.get("is_user", False)
            sources = msg_data.get("sources", [])

            message(msg, is_user=is_user, key=str(i))

            # Show sources if this is an assistant message with sources
            if not is_user and sources:
                display_sources_expander(sources, i)

    st.session_state["thinking_spinner"] = st.empty()


def update_highlights(sources: List[Dict]):
    """
    Update PDF viewer highlights based on retrieved sources

    Args:
        sources: List of source dictionaries with coordinates
    """
    highlights = []

    for source in sources:
        citation_num = source.get("citation_number")
        coordinates = source.get("coordinates", [])

        for coord in coordinates:
            highlights.append({
                "page": coord["page"],
                "bbox": coord["bbox"],
                "citation_number": citation_num,
                "text": coord.get("text", "")
            })

    # Update session state
    st.session_state["highlight_annotations"] = highlights


def process_input():
    if (
        st.session_state["user_input"]
        and len(st.session_state["user_input"].strip()) > 0
    ):
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            response = st.session_state["assistant"].ask(user_text)

        # Handle both old string format and new dict format
        if isinstance(response, dict):
            agent_text = response.get("answer", "")
            sources = response.get("sources", [])

            # Update highlights from sources
            update_highlights(sources)

            # Add messages with new format
            st.session_state["messages"].append({
                "content": user_text,
                "is_user": True
            })
            st.session_state["messages"].append({
                "content": agent_text,
                "is_user": False,
                "sources": sources
            })
        else:
            # Legacy format (string response)
            st.session_state["messages"].append((user_text, True))
            st.session_state["messages"].append((response, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # Store PDF for viewer
        st.session_state["current_pdf_path"] = file_path
        file.seek(0)  # Reset file pointer
        st.session_state["current_pdf_bytes"] = file.read()

        with st.session_state["ingestion_spinner"], st.spinner(
            f"Ingesting {file.name}"
        ):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        st.session_state["messages"].append({
            "content": f"Ingested {file.name} in {t1 - t0:.2f} seconds",
            "is_user": False
        })
        # Don't delete the file immediately, we need it for the PDF viewer
        # os.remove(file_path)


def render_pdf_viewer():
    """Render PDF with highlight annotations"""
    if not st.session_state.get("current_pdf_bytes"):
        st.info("Upload a PDF to view it here")
        return

    # Prepare annotations for highlighting
    annotations = []

    for highlight in st.session_state.get("highlight_annotations", []):
        # Note: streamlit-pdf-viewer uses 1-based page numbering
        # pdfplumber uses 0-based, so we add 1
        annotation = {
            "page": highlight["page"] + 1,  # Convert from 0-based to 1-based
            "x": highlight["bbox"][0],
            "y": highlight["bbox"][1],
            "width": highlight["bbox"][2] - highlight["bbox"][0],
            "height": highlight["bbox"][3] - highlight["bbox"][1],
            "color": "red",  # Changed to red for better visibility
            "opacity": 0.6,  # Increased opacity for better visibility
        }
        annotations.append(annotation)

    # Debug: Show annotation info
    if annotations:
        with st.expander("🔍 Debug: Highlight Info", expanded=False):
            st.write(f"Total highlights: {len(annotations)}")
            for i, ann in enumerate(annotations[:3]):  # Show first 3
                st.write(f"Highlight {i+1}: Page {ann['page']}, Bbox: ({ann['x']:.1f}, {ann['y']:.1f}, {ann['width']:.1f}x{ann['height']:.1f})")

    # Render PDF viewer
    try:
        pdf_viewer(
            input=st.session_state["current_pdf_bytes"],
            width=700,
            height=900,
            annotations=annotations,
            render_text=True,
        )
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")
        st.info("PDF viewer requires streamlit-pdf-viewer package")


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDF()
    if "current_pdf_path" not in st.session_state:
        st.session_state["current_pdf_path"] = None
    if "current_pdf_bytes" not in st.session_state:
        st.session_state["current_pdf_bytes"] = None
    if "highlight_annotations" not in st.session_state:
        st.session_state["highlight_annotations"] = []


def page():
    initialize_session_state()

    st.header("ChatPDF")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    # Two-column layout if PDF is loaded
    if st.session_state.get("current_pdf_bytes"):
        col1, col2 = st.columns([1, 1])  # Equal width columns

        with col1:
            display_messages()
            st.text_input("Message", key="user_input", on_change=process_input)

        with col2:
            st.subheader("PDF Document")
            render_pdf_viewer()
    else:
        # No PDF loaded, show only chat
        display_messages()
        st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()