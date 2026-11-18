"""Node: reword the search query and count the attempt."""

import logging

from northstar_rag.config import settings
from northstar_rag.graph.state import GraphState
from northstar_rag.llm import rewrite_question

logger = logging.getLogger(__name__)


def rewrite_node(state: GraphState) -> dict:
    """Reword the search query and add 1 to the retry counter."""
    attempt = state["rewrite_count"] + 1
    logger.info("No relevant chunks - rewrite %d of %d", attempt, settings.max_rewrites)
    return {
        "search_query": rewrite_question(state["search_query"]),
        "rewrite_count": attempt,
    }
