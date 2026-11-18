"""Connect the nodes into the LangGraph workflows.

Main graph (built by build_graph):

    START -> router --"retrieval_agent"--> retrieval_agent -> END   (NorthStar Bank questions)
                    --"web_search"-------> web_search ------> END   (everything else)

retrieval_agent sub-graph (built by build_retrieval_agent) - the RAG loop with retries:

    START -> classify -> retrieve -> grade --"generate"--> generate -> END
                            ^          |
                            |          +----"rewrite"---> rewrite --+
                            +---------------------------------------+   (search again)
                                       |
                                       +----"fallback"--> fallback -> END
"""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from northstar_rag.config import settings
from northstar_rag.graph.nodes import (
    classify_node,
    fallback_node,
    generate_node,
    grade_node,
    retrieve_node,
    rewrite_node,
    router_node,
    web_search_node,
)
from northstar_rag.graph.state import GraphState, initial_state
from northstar_rag.llm import Route

# The three places the RAG loop can go after grading
AfterGrading = Literal["generate", "rewrite", "fallback"]


# ---------- Routing functions (used by conditional edges) ----------

def decide_route(state: GraphState) -> Route:
    """Return the router's choice; it is also the name of the next node."""
    if state["route"] is None:
        raise ValueError("decide_route ran before router_node set 'route'")
    return state["route"]


def decide_after_grading(state: GraphState) -> AfterGrading:
    """Choose the next step once the grader has filtered the chunks.

    - relevant chunks found       -> "generate" an answer from them
    - none found, retries left    -> "rewrite" the query and search again
    - none found, no retries left -> "fallback" (polite "not found" message)

    With max_rewrites = 3 the guides are searched at most 4 times
    (the first search + 3 rewrites).
    """
    if state["documents"]:
        return "generate"
    if state["rewrite_count"] < settings.max_rewrites:
        return "rewrite"
    return "fallback"


# ---------- Graph builders ----------

def build_retrieval_agent() -> CompiledStateGraph:
    """Build the RAG loop: classify -> retrieve -> grade -> generate / rewrite / fallback."""
    graph = StateGraph(GraphState)

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", decide_after_grading, ["generate", "rewrite", "fallback"])
    graph.add_edge("rewrite", "retrieve")  # retry loop: search again with the new query
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def build_graph() -> CompiledStateGraph:
    """Build the main graph: router -> retrieval_agent (bank guides) or web_search (Tavily)."""
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    # A compiled graph can be used as a node. It shares GraphState with the
    # main graph, so its result flows straight back into the main state.
    graph.add_node("retrieval_agent", build_retrieval_agent())
    graph.add_node("web_search", web_search_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", decide_route, ["retrieval_agent", "web_search"])
    graph.add_edge("retrieval_agent", END)
    graph.add_edge("web_search", END)

    return graph.compile()


def ask(question: str) -> dict[str, Any]:
    """Run one question through the whole graph and return the final state.

    Convenience helper: it compiles the graph on every call. Long-running
    code (like the CLI) should call build_graph() once and reuse it.
    """
    return build_graph().invoke(initial_state(question))
