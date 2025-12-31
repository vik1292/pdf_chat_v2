# ChatPDF - RAG-based PDF Question Answering

A Streamlit-based web application that enables conversational interaction with PDF documents using Retrieval-Augmented Generation (RAG). Upload PDFs and ask questions about their content using natural language.

## Features

- **PDF Upload**: Support for multiple PDF document uploads
- **Intelligent Question Answering**: Uses RAG to provide accurate, context-aware answers
- **Local LLM**: Powered by Ollama's Mistral 7B model for privacy and offline capability
- **Vector Search**: ChromaDB vector store with similarity-based retrieval
- **Interactive Chat Interface**: User-friendly chat UI built with Streamlit

## Architecture

The application uses a RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Ingestion** ([rag.py:31-43](rag.py#L31-L43))
   - PDFs are loaded and split into chunks (1024 characters with 100-character overlap)
   - Chunks are embedded using FastEmbed
   - Embeddings stored in ChromaDB vector database

2. **Query Processing** ([rag.py:50-54](rag.py#L50-L54))
   - User questions are embedded and matched against document chunks
   - Top 3 most relevant chunks retrieved (similarity threshold: 0.5)
   - Context passed to Mistral 7B for answer generation

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
pdfchat_rag/
├── app.py              # Streamlit web application
├── rag.py              # RAG implementation with LangChain
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Key Components

### app.py
Main Streamlit application providing:
- File upload interface ([app.py:66-73](app.py#L66-L73))
- Chat message display ([app.py:12-16](app.py#L12-L16))
- User input processing ([app.py:19-30](app.py#L19-L30))
- Document ingestion workflow ([app.py:32-55](app.py#L32-L55))

### rag.py
Core RAG logic using LangChain:
- `ChatPDF` class for managing the RAG pipeline
- Vector store initialization with ChromaDB
- Document chunking and embedding
- Similarity-based retrieval
- LLM-based answer generation

## Configuration

### Model Configuration
The default LLM is Mistral 7B. To use a different model, modify [rag.py:18](rag.py#L18):

```python
self.model = ChatOllama(model="your-model-name")
```

### Chunk Settings
Adjust document chunking in [rag.py:19](rag.py#L19):

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,      # Size of each chunk
    chunk_overlap=100     # Overlap between chunks
)
```

### Retrieval Settings
Modify retrieval parameters in [rag.py:37-42](rag.py#L37-L42):

```python
self.retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,                    # Number of chunks to retrieve
        "score_threshold": 0.5,    # Minimum similarity score
    },
)
```

## Dependencies

- **langchain**: Framework for LLM applications
- **langchain-community**: Community integrations for LangChain
- **streamlit**: Web application framework
- **streamlit-chat**: Chat UI components
- **chromadb**: Vector database
- **fastembed**: Fast embedding generation
- **pypdf**: PDF document processing

See [requirements.txt](requirements.txt) for complete list.

## Limitations

- Requires Ollama to be running locally
- Answer quality depends on PDF text extraction quality
- Memory usage increases with number of uploaded documents
- ChromaDB vector store is in-memory (not persisted between sessions)

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
- Reduce chunk retrieval count (`k` parameter)
- Use a smaller/faster model
- Ensure sufficient RAM for the model

## Future Enhancements

- Persistent vector store
- Multiple LLM provider support
- Document source citation
- Conversation history persistence
- Advanced filtering options
- Export chat history

## License

This project is provided as-is for educational and personal use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Built with [LangChain](https://langchain.com)
- UI powered by [Streamlit](https://streamlit.io)
- LLM inference via [Ollama](https://ollama.ai)
