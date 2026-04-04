# tools/web_search.py
# ─────────────────────────────────────────────────────────────────
# Fix: BaseTool now imports from 'crewai.tools' not 'crewai_tools'
# ─────────────────────────────────────────────────────────────────

from crewai.tools import BaseTool          # ← FIXED import
from tavily import TavilyClient
from config.settings import TAVILY_API_KEY


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Search the web for current information on any topic. "
        "Input: a search query string. "
        "Output: top search results with titles, URLs, and summaries."
    )

    def _run(self, query: str) -> str:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )
        results = []
        for r in response.get("results", []):
            results.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Summary: {r['content'][:300]}...\n"
            )
        return "\n---\n".join(results) if results else "No results found."