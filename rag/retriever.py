# rag/retriever.py
# ─────────────────────────────────────────────────────────────────
# Exposes a simple retrieve() function that agents can call.
# Takes a query string → returns top-k relevant document chunks.
#
# Flow: query string → embed query → similarity search → top chunks
# ─────────────────────────────────────────────────────────────────

from pathlib import Path
from rag.embedder import load_vector_store, build_vector_store
from rag.ingestor import load_documents, chunk_documents
from config.settings import VECTOR_STORE_PATH, TOP_K_RESULTS


def get_retriever():
    """
    Returns a LangChain retriever object.
    Loads from disk if vector store exists, builds it if not.
    """
    store_path = Path(VECTOR_STORE_PATH)

    if store_path.exists():
        # Vector store already built — just load it
        vector_store = load_vector_store()
    else:
        # First run — ingest documents and build the store
        print("[Retriever] No vector store found. Building from data/raw/...")
        docs   = load_documents()
        chunks = chunk_documents(docs)
        vector_store = build_vector_store(chunks)

    # as_retriever() wraps FAISS in LangChain's Retriever interface
    # search_kwargs={"k": TOP_K_RESULTS} controls how many chunks return
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )


def retrieve(query: str) -> str:
    """
    Main function called by agents.
    Returns relevant document chunks as a formatted string.
    """
    retriever = get_retriever()

    # Run similarity search — returns list of Document objects
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found in the knowledge base."

    # Format each chunk with its source metadata
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        results.append(
            f"[Chunk {i} | Source: {source}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)