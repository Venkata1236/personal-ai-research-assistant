# rag/ingestor.py
# ─────────────────────────────────────────────────────────────────
# Handles loading documents from disk and splitting them
# into chunks that can be embedded into the vector store.
#
# Flow: raw file → Document objects → chunked Documents
# ─────────────────────────────────────────────────────────────────

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,       # For PDF files
    TextLoader,        # For .txt / .md files
    DirectoryLoader,   # Loads all files in a folder
)
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(data_path: str = "./data/raw") -> list:
    """
    Load all documents from the raw data folder.
    Returns a list of LangChain Document objects.
    Each Document has: page_content (str) + metadata (dict).
    """
    path = Path(data_path)
    docs = []

    # Walk through all files in the data directory
    for file in path.glob("**/*"):
        if file.suffix == ".pdf":
            # PyPDFLoader splits a PDF into one Document per page
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())

        elif file.suffix in [".txt", ".md"]:
            # TextLoader loads the entire file as one Document
            loader = TextLoader(str(file), encoding="utf-8")
            docs.extend(loader.load())

    print(f"[Ingestor] Loaded {len(docs)} document pages from {data_path}")
    return docs


def chunk_documents(docs: list) -> list:
    """
    Split large documents into smaller overlapping chunks.
    Why overlap? So context at chunk boundaries isn't lost.

    CHUNK_SIZE    = max characters per chunk (from .env)
    CHUNK_OVERLAP = characters shared between adjacent chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try splitting on paragraphs first, then sentences, then words
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    print(f"[Ingestor] Split into {len(chunks)} chunks")
    return chunks