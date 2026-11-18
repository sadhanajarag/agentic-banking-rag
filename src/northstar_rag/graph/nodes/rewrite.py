"""Node: reword the search query and count the attempt."""

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import rewrite_question


def rewrite_node(state: GraphState) -> dict:
    """Reword the search query and add 1 to the retry counter."""
    return {
        "search_query": rewrite_question(state["search_query"]),
        "rewrite_count": state["rewrite_count"] + 1,
    }
