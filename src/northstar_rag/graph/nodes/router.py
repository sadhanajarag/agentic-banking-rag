"""Node: decide between the retrieval agent and web search."""

import logging

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import route_question

logger = logging.getLogger(__name__)


def router_node(state: GraphState) -> dict:
    """Ask the LLM whether the question needs the bank guides or a web search."""
    route = route_question(state["question"])
    logger.info("Route: %s", route)
    return {"route": route}
