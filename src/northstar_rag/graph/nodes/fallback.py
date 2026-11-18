"""Node: nothing relevant found after all rewrites -> polite fallback."""

import logging

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import FALLBACK_ANSWER

logger = logging.getLogger(__name__)


def fallback_node(state: GraphState) -> dict:
    """Return the polite "not found in the guides" message."""
    logger.info("No relevant chunks after %d rewrites - fallback answer", state["rewrite_count"])
    return {"answer": FALLBACK_ANSWER}
