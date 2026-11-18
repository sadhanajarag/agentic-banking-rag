"""The shared state that flows through every node of the graph."""

from typing import TypedDict

from langchain_core.documents import Document

from northstar_rag.llm import Domain, Route


class GraphState(TypedDict):
    """Data shared by all nodes.

    A node never edits the state directly: it returns a dict with only the
    fields it changes, and LangGraph merges that dict into the state.
    """

    question: str               # the user's original question (never changed)
    route: Route | None         # set by the router; None until the router runs
    search_query: str           # what we search with; the rewriter changes this
    domain: Domain              # which guide to search ("all" = every guide)
    documents: list[Document]   # retrieve: top chunks -> grade: only the relevant ones
    rewrite_count: int          # rewrites done so far (stops at settings.max_rewrites)
    answer: str                 # final answer, or a fallback message


def initial_state(question: str) -> GraphState:
    """Return the starting state for a new question."""
    return GraphState(
        question=question,
        route=None,
        search_query=question,
        domain="all",
        documents=[],
        rewrite_count=0,
        answer="",
    )
