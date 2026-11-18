"""Query rewriter: reword a question so the next search finds better chunks."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from northstar_rag.llm.client import get_llm

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You improve search queries for NorthStar Bank's customer guides "
     "(credit cards, personal loans, digital banking and accounts).\n"
     "The last search did not find useful information. "
     "Rewrite the question so a search finds better matches:\n"
     "- keep the original meaning\n"
     "- use banking terms the guides are likely to use "
     "(for example: fee, APR, grace period, limit)\n"
     "- write product names in full\n"
     "Reply with the rewritten question only."),
    ("human", "{question}"),
])


def rewrite_question(question: str) -> str:
    """Return a reworded version of the question (the retry counter is kept by the graph)."""
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"question": question}).strip()
