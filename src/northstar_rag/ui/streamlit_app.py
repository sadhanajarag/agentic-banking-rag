"""Streamlit chat UI (presentation only - all logic lives in ChatService).

Run:  uv run streamlit run src/northstar_rag/ui/streamlit_app.py
"""

import logging

import streamlit as st

from northstar_rag.logging_config import setup_logging
from northstar_rag.service import ChatService, is_valid_session_id, new_session_id

logger = logging.getLogger(__name__)

ROUTE_LABELS = {"retrieval_agent": "📚 NorthStar Bank guides", "web_search": "🌐 Web search"}


@st.cache_resource  # built once and shared by all browser tabs (graph + SQLite memory)
def get_service() -> ChatService:
    """Create the chat service the first time the app runs."""
    setup_logging()
    return ChatService()


def show(text: str) -> None:
    """Show Markdown safely: Streamlit treats $...$ as math, which breaks '$30 ... $41'."""
    st.markdown(text.replace("$", r"\$"))


def current_session_id() -> str:
    """Session id from the URL (?session=...), so a refresh or saved link keeps the chat."""
    session_id = st.query_params.get("session", "")
    if not is_valid_session_id(session_id):
        session_id = new_session_id()
        st.query_params["session"] = session_id
    return session_id


st.set_page_config(page_title="NorthStar Bank Assistant", page_icon="🏦")
service = get_service()
session_id = current_session_id()

with st.sidebar:
    st.header("🏦 NorthStar Bank")
    st.caption("Ask about credit cards, personal loans, accounts and digital banking.")
    if st.button("➕ New chat", use_container_width=True):
        st.query_params["session"] = new_session_id()
        st.rerun()
    st.caption(f"Session: {session_id[:8]}")

st.title("NorthStar Bank Assistant")

# Replay the conversation saved in SQLite
for message in service.history(session_id):
    with st.chat_message("user" if message.type == "human" else "assistant"):
        show(message.content)

if question := st.chat_input("e.g. What is the late payment fee on my credit card?"):
    with st.chat_message("user"):
        show(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching..."):
                result = service.ask(session_id, question)
        except ValueError as error:  # input problems: the message is meant for the user
            st.warning(str(error))
            st.stop()
        except Exception:  # anything else: full details to the log, short message to the user
            logger.exception("Failed to answer question")
            st.error("Sorry, something went wrong. Please try again.")
            st.stop()

        show(result.answer)
        st.caption(ROUTE_LABELS.get(result.route, result.route))
        if result.sources:
            with st.expander("Sources"):
                for label in result.sources:
                    st.text(label)
