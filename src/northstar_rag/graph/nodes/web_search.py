"""Node: search the web with the Tavily tool, then answer from the results."""

import logging

from northstar_rag.graph.state import GraphState
from northstar_rag.llm import generate_web_answer
from northstar_rag.tools import web_search

logger = logging.getLogger(__name__)

WEB_SEARCH_ERROR = "Sorry, web search isn't available right now. Please try again later."


def web_search_node(state: GraphState) -> dict:
    """Search the web for the question and answer from the results.

    If the search itself fails (bad key, network down, Tavily outage), return a
    polite message instead of crashing the whole graph.
    """
    try:
        results = web_search.invoke({"query": state["question"]})
    except Exception as error:  # noqa: BLE001 - any search failure gets the same message
        logger.warning("Web search failed: %s", error)
        return {"answer": WEB_SEARCH_ERROR}
    return {"answer": generate_web_answer(state["question"], results)}
