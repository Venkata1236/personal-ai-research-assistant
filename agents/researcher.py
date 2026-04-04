# agents/researcher.py — fix RAGTool import too
from crewai.tools import BaseTool          # ← FIXED import

class RAGTool(BaseTool):
    name: str = "RAG Knowledge Base Search"
    description: str = (
        "Search the internal knowledge base of uploaded documents. "
        "Input: a search query string. "
        "Output: relevant document chunks with source references."
    )

    def _run(self, query: str) -> str:
        from rag.retriever import retrieve
        return retrieve(query)