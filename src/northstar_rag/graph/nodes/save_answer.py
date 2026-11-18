"""Node: add the final answer to the conversation history."""

from langchain_core.messages import AIMessage

from northstar_rag.graph.state import GraphState


def save_answer_node(state: GraphState) -> dict:
    """Append the answer as an AI message (one place for every route)."""
    return {"messages": [AIMessage(content=state["answer"])]}
