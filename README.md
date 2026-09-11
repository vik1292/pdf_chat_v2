# ChatPDF - RAG-based PDF Question Answering

A Streamlit-based web application that enables conversational interaction with PDF documents using Retrieval-Augmented Generation (RAG). Upload PDFs and ask questions about their content using natural language.

## Features

- **PDF Upload**: Multiple PDFs at once; every source names the document it came from
- **Hybrid Retrieval**: BM25 keyword search fused with cosine vector search, then reranked by a local cross-encoder
- **Precise Source Highlighting**: The exact sentences that answered the question are outlined line by line on the right page, in a color per citation
- **Cited Sources Only**: Only chunks the answer actually cites are highlighted; every retrieved chunk is still listed with its reranker score
- **Local LLM**: Ollama Mistral 7B, fully offline once models are cached
- **Interactive Chat Interface**: Native Streamlit chat with a "Show in PDF" button per citation

## Architecture

1. **Ingestion** (`pdf_extract.py`, `rag.py`)
   - pdfplumber extracts every word with coordinates; words get line indices.
   - Each page is split into ~500-character chunks that end on sentence boundaries. A chunk stores its page and word range, so its location is a lookup, not a text search.
   - Chunks are embedded with fastembed (bge-small) into an in-memory Chroma cosine collection and indexed with BM25. The collection is unique per session and deleted when documents are cleared.

2. **Question answering** (`retrieval.py`, `rag.py`)
   - The original question is searched by vector similarity and BM25; results are fused with reciprocal rank fusion.
   - A local cross-encoder (ms-marco MiniLM) reranks the top 20 candidates; the best 5 go to the LLM with their neighboring chunks as context.
   - The same cross-encoder scores each sentence of a retrieved chunk against the question to pick the evidence sentences.
   - Mistral 7B (Ollama) answers with `[n]` citations; only cited sources are highlighted.

3. **Display** (`app.py`)
   - One outlined box per line: solid for evidence sentences, dashed for the rest of the chunk, colored per citation.
   - The viewer switches to the cited document and scrolls to the cited page. "Show in PDF" isolates one citation.

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Mistral 7B model pulled in Ollama

### Install Ollama and Model

```bash
# Install Ollama (visit https://ollama.ai for installation instructions)

# Pull the Mistral 7B model
ollama pull mistral:7b
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdfchat_rag
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload one or more PDF documents using the file uploader

4. Wait for the ingestion process to complete

5. Ask questions about your documents in the chat interface

## Project Structure

```
pdf_chat_v2/
├── app.py                # Streamlit UI: chat, sources panel, PDF viewer with highlights
├── rag.py                # ChatPDF orchestrator: ingest, ask, clear
├── retrieval.py          # Hybrid index (BM25 + Chroma cosine), RRF, reranker, evidence selection
├── pdf_extract.py        # pdfplumber words with coordinates, line grouping, sentence-aware chunking
├── tests/                # pytest suite (unit tests use fakes; one opt-in integration test)
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Runtime + pytest + reportlab (test PDFs)
└── README.md
```

## Configuration

All knobs are constructor arguments or function defaults:

```python
# rag.py
ChatPDF(model_name="mistral:7b", candidates=20, top_n=5)

# pdf_extract.py
chunk_words(words, doc_id, page, target_chars=500, max_chars=800)

# retrieval.py
FastEmbedder(model_name="BAAI/bge-small-en-v1.5")
FastEmbedReranker(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
select_evidence(scores, margin=2.0, max_items=3)
```

`ChatPDF` also accepts `llm`, `embedder` and `reranker` instances, which is how the tests inject fakes.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                          # unit tests: no models, no Ollama
PDFCHAT_INTEGRATION=1 python -m pytest -m integration -q     # real embedder + reranker
```

## Dependencies

- **streamlit** / **streamlit-pdf-viewer**: UI and annotated PDF rendering
- **langchain-ollama**: `ChatOllama` client for the local LLM
- **chromadb**: In-memory cosine vector index
- **fastembed**: bge-small embeddings and the ms-marco cross-encoder reranker
- **rank-bm25**: Keyword search
- **pdfplumber**: Word-level text extraction with coordinates

Version pins worth knowing: `fastembed<0.8` requires `pillow<12` on Python 3.13, and `pdfplumber` is pinned to 0.11.9 because 0.11.10 requires pillow 12. See [requirements.txt](requirements.txt).

## Limitations

- Requires Ollama to be running locally
- Scanned PDFs without a text layer produce no words and therefore no chunks or highlights
- Sentence detection is punctuation based, so bullet lists without terminal punctuation become one sentence
- Everything is in memory and lost when the session ends

## Troubleshooting

### "Connection Error" when asking questions
Ensure Ollama is running:
```bash
ollama serve
```

### "Model not found" error
Pull the Mistral model:
```bash
ollama pull mistral:7b
```

### Slow response times
- Lower `candidates` or `top_n` on `ChatPDF`
- Use a smaller/faster Ollama model via `model_name`
- The first question after startup downloads the reranker model (about 90 MB) once

## Future Enhancements

- Persistent vector store
- Multiple LLM provider support
- Conversation history in the prompt
- Export chat history

## License

This project is provided as-is for educational and personal use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Built with [LangChain](https://langchain.com)
- UI powered by [Streamlit](https://streamlit.io)
- LLM inference via [Ollama](https://ollama.ai)
