"""The shared state that flows through every node of the graph."""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from northstar_rag.llm import Domain, Route


class GraphState(TypedDict):
    """Data shared by all nodes.

    A node never edits the state directly: it returns a dict with only the
    fields it changes, and LangGraph merges that dict into the state.
    """

    # The whole conversation. add_messages APPENDS new messages instead of replacing
    # the list, and the checkpointer saves it per session between questions.
    messages: Annotated[list[AnyMessage], add_messages]
    question: str               # this turn's question (made standalone by contextualize)
    route: Route | None         # set by the router; None until the router runs
    search_query: str           # what we search with; the rewriter changes this
    domain: Domain              # which guide to search ("all" = every guide)
    documents: list[Document]   # retrieve: top chunks -> grade: only the relevant ones
    rewrite_count: int          # rewrites done so far (stops at settings.max_rewrites)
    answer: str                 # final answer, or a fallback message


def initial_state(question: str) -> GraphState:
    """Return the input for one turn.

    Every per-turn field is reset; the new user message is appended to the
    saved conversation (because of add_messages).
    """
    return GraphState(
        messages=[HumanMessage(content=question)],
        question=question,
        route=None,
        search_query=question,
        domain="all",
        documents=[],
        rewrite_count=0,
        answer="",
    )
