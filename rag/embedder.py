# rag/embedder.py
# ─────────────────────────────────────────────────────────────────
# Converts document chunks into vector embeddings
# and saves them to a FAISS vector store on disk.
#
# Flow: chunks → embeddings → FAISS index saved to disk
# ─────────────────────────────────────────────────────────────────

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import VECTOR_STORE_PATH


# HuggingFace embedding model — runs locally, no API key needed.
# "all-MiniLM-L6-v2" is fast and good enough for most RAG use cases.
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vector_store(chunks: list) -> FAISS:
    """
    Take document chunks, embed them, and save to disk.
    Returns the FAISS vector store object.
    """
    print(f"[Embedder] Embedding {len(chunks)} chunks...")

    # FAISS.from_documents() does two things:
    # 1. Calls the embedding model on each chunk's text
    # 2. Stores the resulting vectors in a FAISS index
    vector_store = FAISS.from_documents(chunks, EMBEDDING_MODEL)

    # Persist to disk so we don't re-embed every time the app runs
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"[Embedder] Vector store saved to {VECTOR_STORE_PATH}")

    return vector_store


def load_vector_store() -> FAISS:
    """
    Load a previously built vector store from disk.
    Call this on app startup instead of re-embedding every time.
    """
    print(f"[Embedder] Loading vector store from {VECTOR_STORE_PATH}")
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        EMBEDDING_MODEL,
        allow_dangerous_deserialization=True  # required by FAISS
    )
    
    