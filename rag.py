from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
import pdfplumber
from typing import List, Dict


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    text_positions = None
    current_pdf_path = None

    def __init__(self):
        self.model = ChatOllama(model="mistral:7b")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are a helpful assistant for answering questions about documents. Follow these instructions carefully:

            INSTRUCTIONS:
            1. Use ONLY the information provided in the Context below to answer questions
            2. If the Context contains relevant information, provide a detailed, specific answer with exact details from the document
            3. Include citation markers [1], [2], [3] for each piece of information from different source chunks
            4. If the Context does NOT contain the answer, respond with: "I don't have information about that in the provided documents."
            5. NEVER make up information or provide generic answers - only use what's explicitly stated in the Context
            6. Be thorough - if multiple sources discuss the topic, synthesize the information and cite all relevant sources

            Context (from documents):
            {context}

            Question: {question}

            Answer with citations: [/INST]
            """
        )

    def expand_query(self, query: str) -> str:
        """
        Use LLM to expand user query with related terms and synonyms

        Args:
            query: Original user query

        Returns:
            Expanded query with additional search terms
        """
        expansion_prompt = f"""Given this user question, generate a list of 5-10 related keywords and synonyms that would help find relevant information in a document. Include both formal and informal terms.

Question: {query}

Output ONLY the keywords separated by spaces, no explanations:"""

        try:
            # Use the LLM to generate expansion terms
            expansion_result = self.model.invoke(expansion_prompt)

            # Extract just the text content
            if hasattr(expansion_result, 'content'):
                expansion_terms = expansion_result.content.strip()
            else:
                expansion_terms = str(expansion_result).strip()

            # Clean up the expansion (remove newlines, extra spaces)
            expansion_terms = ' '.join(expansion_terms.split())

            # Combine original query with expansions
            expanded_query = f"{query} {expansion_terms}"
            print(f"Query expansion: {query} -> {expansion_terms[:100]}...")

            return expanded_query
        except Exception as e:
            print(f"Query expansion failed: {e}, using original query")
            return query

    def extract_text_positions(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """
        Extract text with bounding box coordinates from PDF

        Returns:
            Dict mapping page numbers to list of text objects with positions
        """
        positions = {}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_positions = []

                    # Extract words with coordinates
                    words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False
                    )

                    for word in words:
                        page_positions.append({
                            "text": word["text"],
                            "bbox": (word["x0"], word["top"], word["x1"], word["bottom"]),
                            "page": page_num
                        })

                    positions[page_num] = page_positions
        except Exception as e:
            print(f"Error extracting text positions: {e}")
            positions = {}

        return positions

    def find_chunk_coordinates(
        self,
        chunk_text: str,
        page_num: int,
        text_positions: Dict
    ) -> List[Dict]:
        """
        Find bounding boxes for a text chunk on a specific page

        Args:
            chunk_text: The text to find
            page_num: Page number to search
            text_positions: Output from extract_text_positions()

        Returns:
            List of bounding boxes covering the chunk
        """
        if page_num not in text_positions or not text_positions[page_num]:
            print(f"Warning: Page {page_num} not found in text_positions")
            return []

        page_words = text_positions[page_num]
        # Normalize chunk text: remove extra whitespace and split
        chunk_words = chunk_text.split()

        if not chunk_words:
            print(f"Warning: No words in chunk for page {page_num}")
            return []

        print(f"Searching for {len(chunk_words)} words on page {page_num} (first 5: {chunk_words[:5]})")

        # Find matching word sequences - use first 10 words for better matching
        matches = []
        search_words = chunk_words[:10]  # Search for first 10 words

        for i in range(len(page_words) - len(search_words) + 1):
            # Check if words match (case-insensitive, normalize whitespace)
            match = True

            for j in range(len(search_words)):
                if i + j >= len(page_words):
                    match = False
                    break

                page_word = page_words[i + j]["text"].lower().strip()
                chunk_word = search_words[j].lower().strip()

                if page_word != chunk_word:
                    match = False
                    break

            if match:
                print(f"Match found at word index {i} on page {page_num}")
                # Combine bboxes of all matching words (up to 100 for full context)
                words_to_combine = min(len(chunk_words), 100)
                word_bboxes = [page_words[i + j]["bbox"] for j in range(min(words_to_combine, len(page_words) - i))]

                if word_bboxes:
                    # Create merged bounding box
                    merged_bbox = (
                        min(b[0] for b in word_bboxes),  # x0
                        min(b[1] for b in word_bboxes),  # y0
                        max(b[2] for b in word_bboxes),  # x1
                        max(b[3] for b in word_bboxes),  # y1
                    )

                    matches.append({
                        "bbox": merged_bbox,
                        "page": page_num,
                        "text": chunk_text[:200]  # Truncate for display
                    })
                    break  # Only find first match

        if not matches:
            print(f"Warning: No matches found for chunk on page {page_num}")
            print(f"Page has {len(page_words)} words, first few: {[w['text'] for w in page_words[:10]]}")

        return matches

    def ingest(self, pdf_file_path: str):
        """Load PDF and extract text + positions"""
        # Store PDF path for later reference
        self.current_pdf_path = pdf_file_path

        # Load documents
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        # Extract text positions
        self.text_positions = self.extract_text_positions(pdf_file_path)

        # Create vector store
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,  # Increased from 3 to 5 for better recall
                "score_threshold": 0.3,  # Lowered from 0.5 to 0.3 for better recall
            },
        )

        # Create chain with RunnableParallel to preserve context
        self.chain = RunnableParallel(
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
        ).assign(
            answer=self.prompt | self.model | StrOutputParser()
        )

    def ask(self, query: str) -> dict:
        """
        Query the RAG system and return answer with source metadata

        Returns:
            dict: {
                "answer": str,
                "sources": List[dict] with metadata and coordinates
            }
        """
        if not self.chain:
            return {
                "answer": "Please, upload a PDF document first.",
                "sources": []
            }

        # Expand query for better retrieval
        expanded_query = self.expand_query(query)

        # Retrieve documents
        result = self.chain.invoke(expanded_query)

        # Extract source documents from context
        source_docs = result.get("context", [])

        # Log retrieval results for debugging
        print(f"\n{'='*50}")
        print(f"Original Query: {query}")
        print(f"Expanded Query: {expanded_query[:200]}...")
        print(f"Retrieved {len(source_docs)} documents:")
        for idx, doc in enumerate(source_docs, 1):
            page_num = doc.metadata.get('page', '?')
            score = doc.metadata.get('score', 'N/A')
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"\n[{idx}] Page {page_num} (Score: {score})")
            print(f"    Preview: {preview}...")
        print(f"{'='*50}\n")

        # Format sources with coordinates
        sources = []
        for idx, doc in enumerate(source_docs, start=1):
            page_num = doc.metadata.get("page", 0)
            chunk_text = doc.page_content

            # Find coordinates for this chunk
            coordinates = []
            if self.text_positions:
                coordinates = self.find_chunk_coordinates(
                    chunk_text=chunk_text,
                    page_num=page_num,
                    text_positions=self.text_positions
                )

            sources.append({
                "citation_number": idx,
                "text": chunk_text,
                "page": page_num,
                "page_label": str(page_num + 1),
                "source": doc.metadata.get("source", ""),
                "coordinates": coordinates,
                "score": doc.metadata.get("score", 0.0),
            })

        return {
            "answer": result.get("answer", ""),
            "sources": sources
        }

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.text_positions = None
        self.current_pdf_path = None


