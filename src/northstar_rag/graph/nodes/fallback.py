"""Node: nothing relevant found after all rewrites -> polite fallback."""

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import FALLBACK_ANSWER


def fallback_node(state: GraphState) -> dict:
    """Return the polite "not found in the guides" message."""
    return {"answer": FALLBACK_ANSWER}
