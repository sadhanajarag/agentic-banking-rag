from northstar_rag.llm import grade_documents


def test_empty_docs_returns_empty():
    assert grade_documents("any question", []) == []
