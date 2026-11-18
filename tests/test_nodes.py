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


def test_contextualize_skips_llm_without_history(monkeypatch):
    from northstar_rag.graph.nodes import contextualize

    def must_not_be_called(*args):
        raise AssertionError("LLM called for the first question")

    monkeypatch.setattr(contextualize, "contextualize_question", must_not_be_called)
    assert contextualize.contextualize_node(initial_state("annual fee?")) == {}


def test_save_answer_appends_ai_message():
    from northstar_rag.graph.nodes import save_answer_node

    state = initial_state("q") | {"answer": "It is $95."}
    message = save_answer_node(state)["messages"][0]
    assert message.type == "ai"
    assert message.content == "It is $95."
