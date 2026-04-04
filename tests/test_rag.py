# tests/test_rag.py
# Basic tests to verify the RAG pipeline works end-to-end.

from rag.retriever import retrieve


def test_retrieve_returns_string():
    """retrieve() should always return a string, even with no docs."""
    result = retrieve("artificial intelligence")
    assert isinstance(result, str)


def test_retrieve_not_empty():
    """If docs exist, result shouldn't be empty."""
    result = retrieve("machine learning")
    assert len(result) > 0