"""Node: keep only the chunks that help answer the user's question."""

import logging

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import grade_documents

logger = logging.getLogger(__name__)


def grade_node(state: GraphState) -> dict:
    """Replace the retrieved chunks with only the ones relevant to the original question."""
    relevant = grade_documents(state["question"], state["documents"])
    logger.info("Grader kept %d of %d chunks", len(relevant), len(state["documents"]))
    return {"documents": relevant}
