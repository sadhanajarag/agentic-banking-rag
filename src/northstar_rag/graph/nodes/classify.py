"""Node: decide which guide the question belongs to."""

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import classify_domain


def classify_node(state: GraphState) -> dict:
    """Ask the LLM which guide the question is about."""
    return {"domain": classify_domain(state["question"])}
