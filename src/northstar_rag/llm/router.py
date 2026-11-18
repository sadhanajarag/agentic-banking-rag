"""Router: send banking questions to the retrieval agent, everything else to web search."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from northstar_rag.llm.client import get_llm

# The router's possible answers. Each one is also the name of a node in the main graph.
Route = Literal["retrieval_agent", "web_search"]


class RouteResult(BaseModel):
    """The only answers the LLM is allowed to give."""

    route: Route = Field(description="Where the question should go")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Route the user's question:\n"
     "- retrieval_agent: questions about NorthStar Bank products and services: "
     "credit cards, personal loans, checking/savings accounts, mobile/online banking, "
     "fees, rates, transfers, policies\n"
     "- web_search: everything else "
     "(general knowledge, news, other companies, not about NorthStar Bank)"),
    ("human", "{question}"),
])


def route_question(question: str) -> Route:
    """Ask the LLM where the question should go."""
    chain = PROMPT | get_llm().with_structured_output(RouteResult)
    return chain.invoke({"question": question}).route
