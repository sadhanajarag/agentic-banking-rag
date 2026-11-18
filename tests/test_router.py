from northstar_rag.graph import builder, decide_route, initial_state


def test_decide_route():
    state = initial_state("q") | {"route": "web_search"}
    assert decide_route(state) == "web_search"


def test_graph_uses_web_search(monkeypatch):
    """Fake router sends the question to a fake web search node (no API calls)."""
    monkeypatch.setattr(builder, "router_node", lambda s: {"route": "web_search"})
    monkeypatch.setattr(builder, "web_search_node", lambda s: {"answer": "web answer"})
    result = builder.ask("Who won the World Cup?")
    assert result["route"] == "web_search"
    assert result["answer"] == "web answer"


def test_web_search_failure_gives_polite_answer(monkeypatch):
    """If Tavily fails, the node returns a message instead of crashing the graph."""
    from types import SimpleNamespace

    from northstar_rag.graph.nodes import web_search as node_module

    def broken_search(_):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(node_module, "web_search", SimpleNamespace(invoke=broken_search))
    result = node_module.web_search_node(initial_state("q"))
    assert result == {"answer": node_module.WEB_SEARCH_ERROR}
