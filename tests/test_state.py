from northstar_rag.graph import initial_state


def test_initial_state():
    state = initial_state("What is the annual fee?")
    assert state["question"] == "What is the annual fee?"
    assert state["route"] is None
    assert state["search_query"] == state["question"]
    assert state["domain"] == "all"
    assert state["documents"] == []
    assert state["rewrite_count"] == 0
    assert state["answer"] == ""
