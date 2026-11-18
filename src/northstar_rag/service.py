"""Application layer: the single entry point UIs use (Streamlit now, FastAPI later).

The UI only knows about ChatService and ChatResult; it never touches the graph,
the retrievers or the LLM directly.
"""

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from northstar_rag.config import settings
from northstar_rag.graph import build_graph, initial_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResult:
    """Everything the UI needs to show one answer."""

    answer: str
    route: str                                         # "retrieval_agent" or "web_search"
    standalone_question: str                           # question after follow-up rewriting
    sources: list[str] = field(default_factory=list)   # "[1] file.pdf > section" (bank answers)


# ---------- Sessions ----------

def new_session_id() -> str:
    """Create a random session id. It is also LangGraph's thread_id, which keys the memory."""
    return str(uuid.uuid4())


def is_valid_session_id(value: str) -> bool:
    """True if value is a UUID (e.g. from the page URL); anything else is rejected."""
    try:
        uuid.UUID(value)
    except (ValueError, TypeError):
        return False
    return True


# ---------- Memory ----------

def create_checkpointer(db_path: Path = settings.chat_db_path) -> SqliteSaver:
    """Open (or create) the SQLite file that stores every session's conversation."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: Streamlit answers each request on a different thread.
    # Sharing one connection is safe because SqliteSaver uses its own lock.
    connection = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(connection)


# ---------- Service ----------

class ChatService:
    """Answer questions through the graph and keep each session's conversation."""

    def __init__(self, graph: CompiledStateGraph | None = None) -> None:
        """Use the given graph (tests), or build the real one with SQLite memory."""
        self._graph = graph or build_graph(checkpointer=create_checkpointer())

    def ask(self, session_id: str, question: str) -> ChatResult:
        """Run one question in a session and return the answer with its route and sources.

        Raises ValueError for empty or too-long questions (safe to show to the user).
        """
        question = question.strip()
        if not question:
            raise ValueError("Please enter a question.")
        if len(question) > settings.max_question_chars:
            raise ValueError(
                f"Please keep questions under {settings.max_question_chars} characters."
            )

        logger.info("Session %s: new question (%d chars)", session_id[:8], len(question))
        final = self._graph.invoke(initial_state(question), config=self._config(session_id))
        return ChatResult(
            answer=final["answer"],
            route=final["route"],
            standalone_question=final["question"],
            sources=source_labels(final["documents"]),
        )

    def history(self, session_id: str) -> list[BaseMessage]:
        """Return the session's messages (empty list for a new session)."""
        snapshot = self._graph.get_state(self._config(session_id))
        return snapshot.values.get("messages", [])

    @staticmethod
    def _config(session_id: str) -> RunnableConfig:
        """LangGraph config: thread_id selects which conversation to load and save."""
        return {"configurable": {"thread_id": session_id}}


def source_labels(docs: list[Document]) -> list[str]:
    """Number the chunks the same way as the generator, so [1] in the answer matches."""
    return [
        f"[{number}] {doc.metadata.get('source')} > {doc.metadata.get('section') or 'n/a'}"
        for number, doc in enumerate(docs, start=1)
    ]
