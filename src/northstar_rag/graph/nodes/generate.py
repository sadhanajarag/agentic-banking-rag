"""Node: write the final answer from the relevant chunks."""

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import generate_answer


def generate_node(state: GraphState) -> dict:
    """Write the final answer from the relevant chunks, with [1], [2] citations."""
    return {"answer": generate_answer(state["question"], state["documents"])}
