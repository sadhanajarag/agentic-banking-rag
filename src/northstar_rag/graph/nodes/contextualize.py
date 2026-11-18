"""Node: make follow-up questions standalone using the conversation history."""

import logging

from northstar_rag.config import settings
from northstar_rag.graph.state import GraphState
from northstar_rag.llm import contextualize_question

logger = logging.getLogger(__name__)


def contextualize_node(state: GraphState) -> dict:
    """Rewrite a follow-up into a standalone question; skip the LLM when there is no history."""
    # messages[-1] is the current question, so the history is everything before it
    history = state["messages"][:-1][-settings.history_messages:]
    if not history:
        return {}
    standalone = contextualize_question(state["question"], history)
    logger.debug("Standalone question: %s", standalone)  # DEBUG: questions may hold personal data
    return {"question": standalone, "search_query": standalone}
