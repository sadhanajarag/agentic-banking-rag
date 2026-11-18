import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from northstar_rag.graph import builder
from northstar_rag.service import (
    ChatService,
    create_checkpointer,
    is_valid_session_id,
    new_session_id,
    source_labels,
)

DOC = Document(page_content="Late fee is $30", metadata={"source": "card.pdf", "section": "Fees"})


@pytest.fixture
def fake_nodes(monkeypatch):
    """Swap every node that calls an API for a fake (bank route, always finds DOC)."""
    monkeypatch.setattr(builder, "contextualize_node", lambda s: {})
    monkeypatch.setattr(builder, "router_node", lambda s: {"route": "retrieval_agent"})
    monkeypatch.setattr(builder, "classify_node", lambda s: {"domain": "all"})
    monkeypatch.setattr(builder, "retrieve_node", lambda s: {"documents": [DOC]})
    monkeypatch.setattr(builder, "grade_node", lambda s: {"documents": [DOC]})
    monkeypatch.setattr(builder, "generate_node", lambda s: {"answer": "It is $30 [1]."})


def test_empty_and_too_long_questions_rejected():
    service = ChatService(graph=object())  # never reached: validation fails first
    with pytest.raises(ValueError):
        service.ask(new_session_id(), "   ")
    with pytest.raises(ValueError):
        service.ask(new_session_id(), "x" * 5000)


def test_memory_is_kept_per_session(fake_nodes):
    service = ChatService(graph=builder.build_graph(checkpointer=InMemorySaver()))
    session = new_session_id()

    result = service.ask(session, "What is the late fee?")
    service.ask(session, "And for the Gold card?")

    assert result.answer == "It is $30 [1]."
    assert result.sources == ["[1] card.pdf > Fees"]
    assert len(service.history(session)) == 4          # 2 questions + 2 answers
    assert service.history(new_session_id()) == []     # other sessions start empty


def test_sqlite_memory_survives_restart(fake_nodes, tmp_path):
    db_path = tmp_path / "chat.sqlite"
    session = new_session_id()

    first_saver = create_checkpointer(db_path)
    ChatService(graph=builder.build_graph(checkpointer=first_saver)).ask(session, "Late fee?")
    first_saver.conn.close()

    # A brand-new service on the same file = the app after a restart
    second_saver = create_checkpointer(db_path)
    restarted = ChatService(graph=builder.build_graph(checkpointer=second_saver))
    assert len(restarted.history(session)) == 2
    second_saver.conn.close()


def test_session_id_validation():
    assert is_valid_session_id(new_session_id())
    assert not is_valid_session_id("not-a-uuid")
    assert not is_valid_session_id("")


def test_source_labels_numbered_like_citations():
    assert source_labels([DOC, DOC]) == ["[1] card.pdf > Fees", "[2] card.pdf > Fees"]
