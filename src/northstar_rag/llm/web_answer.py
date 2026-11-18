"""Write an answer from web search results, with [1], [2] citations."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from northstar_rag.llm.client import get_llm

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using ONLY the numbered web results below.\n"
     "- Cite sources inline as [1], [2].\n"
     "- If the results don't answer it, say you couldn't find a reliable answer.\n"
     "- Be concise.\n"
     "- End with the URLs of the sources you cited.\n\nWeb results:\n{results}"),
    ("human", "{question}"),
])


def generate_web_answer(question: str, results: str) -> str:
    """Answer the question from the web search text."""
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"question": question, "results": results})
