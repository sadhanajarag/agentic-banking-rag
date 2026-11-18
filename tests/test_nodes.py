from northstar_rag.graph import initial_state
from northstar_rag.graph.nodes import classify, fallback_node, rewrite
from northstar_rag.llm import FALLBACK_ANSWER


def test_fallback_node():
    assert fallback_node(initial_state("q")) == {"answer": FALLBACK_ANSWER}


def test_classify_node(monkeypatch):
    monkeypatch.setattr(classify, "classify_domain", lambda q: "credit_card")
    assert classify.classify_node(initial_state("annual fee?")) == {"domain": "credit_card"}


def test_rewrite_node_counts(monkeypatch):
    monkeypatch.setattr(rewrite, "rewrite_question", lambda q: "better " + q)
    result = rewrite.rewrite_node(initial_state("fee?"))
    assert result == {"search_query": "better fee?", "rewrite_count": 1}
