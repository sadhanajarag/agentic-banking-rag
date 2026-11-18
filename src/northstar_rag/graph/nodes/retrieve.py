"""Node: hybrid search + rerank, limited to the classified guide."""

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import domain_to_source
from northstar_rag.retrieval import build_reranked_retriever


def retrieve_node(state: GraphState) -> dict:
    """Search the classified guide (hybrid + rerank) with the current search query."""
    retriever = build_reranked_retriever(domain_to_source(state["domain"]))
    return {"documents": retriever.invoke(state["search_query"])}
