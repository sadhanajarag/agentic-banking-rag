"""Contextualizer: turn a follow-up question into a standalone question using chat history."""

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from northstar_rag.llm.client import get_llm

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history, rewrite the user's latest question so it can be understood "
     "without the history (fill in the product or topic it refers to).\n"
     "Do NOT answer it. If it is already standalone, return it unchanged.\n"
     "Reply with the question only."),
    MessagesPlaceholder("history"),  # the earlier messages are inserted here
    ("human", "{question}"),
])


def contextualize_question(question: str, history: list[BaseMessage]) -> str:
    """Return a standalone version of the question, e.g. 'and the Gold card?' -> full question."""
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"question": question, "history": history}).strip()
