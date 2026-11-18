from langchain_core.documents import Document

from northstar_rag.config import settings
from northstar_rag.graph import builder, decide_after_grading, initial_state
from northstar_rag.llm import FALLBACK_ANSWER

DOC = Document(page_content="Annual fee is $95")


def test_decide_generate_when_docs():
    state = initial_state("q") | {"documents": [DOC]}
    assert decide_after_grading(state) == "generate"


def test_decide_rewrite_when_no_docs():
    assert decide_after_grading(initial_state("q")) == "rewrite"


def test_decide_fallback_after_max_rewrites():
    state = initial_state("q") | {"rewrite_count": settings.max_rewrites}
    assert decide_after_grading(state) == "fallback"


def use_fake_nodes(monkeypatch, grade_result):
    """Swap the real nodes (which call OpenAI) for fakes."""
    monkeypatch.setattr(builder, "router_node", lambda s: {"route": "retrieval_agent"})
    monkeypatch.setattr(builder, "classify_node", lambda s: {"domain": "all"})
    monkeypatch.setattr(builder, "retrieve_node", lambda s: {"documents": [DOC]})
    monkeypatch.setattr(builder, "grade_node", lambda s: {"documents": grade_result})
    monkeypatch.setattr(
        builder, "rewrite_node", lambda s: {"rewrite_count": s["rewrite_count"] + 1}
    )
    monkeypatch.setattr(builder, "generate_node", lambda s: {"answer": "fake answer"})


def test_graph_answers_when_docs_found(monkeypatch):
    use_fake_nodes(monkeypatch, grade_result=[DOC])
    result = builder.ask("q")
    assert result["answer"] == "fake answer"
    assert result["rewrite_count"] == 0


def test_graph_falls_back_after_max_rewrites(monkeypatch):
    use_fake_nodes(monkeypatch, grade_result=[])
    result = builder.ask("q")
    assert result["answer"] == FALLBACK_ANSWER
    assert result["rewrite_count"] == settings.max_rewrites
