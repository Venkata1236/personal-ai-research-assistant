# rag/ingestor.py
# ─────────────────────────────────────────────────────────────────
# Fix: langchain.text_splitter moved to langchain_text_splitters
# Fix: langchain_community.document_loaders replaces langchain loaders
# ─────────────────────────────────────────────────────────────────

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← FIXED
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(data_path: str = "./data/raw") -> list:
    """
    Load all documents from the raw data folder.
    Returns a list of LangChain Document objects.
    """
    path = Path(data_path)
    docs = []

    if not path.exists():
        print(f"[Ingestor] Data path {data_path} does not exist — skipping.")
        return docs

    for file in path.glob("**/*"):
        try:
            if file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))
                docs.extend(loader.load())

            elif file.suffix in [".txt", ".md"]:
                loader = TextLoader(str(file), encoding="utf-8")
                docs.extend(loader.load())

        except Exception as e:
            print(f"[Ingestor] Skipping {file.name}: {e}")

    print(f"[Ingestor] Loaded {len(docs)} document pages from {data_path}")
    return docs


def chunk_documents(docs: list) -> list:
    """
    Split documents into smaller overlapping chunks for embedding.
    """
    if not docs:
        print("[Ingestor] No documents to chunk.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    print(f"[Ingestor] Split into {len(chunks)} chunks")
    return chunks