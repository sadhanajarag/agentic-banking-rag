"""Node: keep only the chunks that help answer the user's question."""

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import grade_documents


def grade_node(state: GraphState) -> dict:
    """Replace the retrieved chunks with only the ones relevant to the original question."""
    return {"documents": grade_documents(state["question"], state["documents"])}
