"""Web search tool (Tavily) for questions outside the bank's guides."""

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from northstar_rag.config import settings


@tool
def web_search(query: str) -> str:
    """Search the web and return numbered results: [1] title (url) + text."""
    search = TavilySearch(
        max_results=settings.web_search_results,
        tavily_api_key=settings.tavily_api_key,
    )
    response = search.invoke({"query": query})
    if "error" in response:  # Tavily returns {"error": ...} instead of raising
        raise RuntimeError(f"Tavily search failed: {response['error']}")
    return "\n\n---\n\n".join(
        f"[{number}] {result['title']} ({result['url']})\n{result['content']}"
        for number, result in enumerate(response["results"], start=1)
    )
