# rag/embedder.py
# ─────────────────────────────────────────────────────────────────
# Switched from HuggingFace to OpenAI embeddings.
# No local model download needed — uses OpenAI API directly.
# ─────────────────────────────────────────────────────────────────

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings          # ← OpenAI embeddings
from config.settings import VECTOR_STORE_PATH, OPENAI_API_KEY

# OpenAI embedding model — text-embedding-3-small is fast + cheap
EMBEDDING_MODEL = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)


def build_vector_store(chunks: list) -> FAISS:
    """
    Embed chunks using OpenAI and save FAISS index to disk.
    """
    print(f"[Embedder] Embedding {len(chunks)} chunks with OpenAI...")

    vector_store = FAISS.from_documents(chunks, EMBEDDING_MODEL)
    vector_store.save_local(VECTOR_STORE_PATH)

    print(f"[Embedder] Vector store saved to {VECTOR_STORE_PATH}")
    return vector_store


def load_vector_store() -> FAISS:
    """Load existing FAISS vector store from disk."""
    print(f"[Embedder] Loading vector store from {VECTOR_STORE_PATH}")
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        EMBEDDING_MODEL,
        allow_dangerous_deserialization=True
    )