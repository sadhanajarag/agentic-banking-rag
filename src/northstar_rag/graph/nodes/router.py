"""Node: decide between the retrieval agent and web search."""

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import route_question


def router_node(state: GraphState) -> dict:
    """Ask the LLM whether the question needs the bank guides or a web search."""
    return {"route": route_question(state["question"])}
