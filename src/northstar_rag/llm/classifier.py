"""Domain classifier: decide which bank guide a question belongs to."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from northstar_rag.llm.client import get_llm

# The classifier's possible answers. "all" means: search every guide.
Domain = Literal["credit_card", "personal_loan", "digital_banking", "all"]

# Domain label -> PDF file name (used as the retriever's source filter)
GUIDES: dict[str, str] = {
    "credit_card": "NorthStar_Bank_Credit_Card_Customer_Guide.pdf",
    "personal_loan": "NorthStar_Bank_Personal_Loan_Customer_Handbook.pdf",
    "digital_banking": "NorthStar_Bank_Digital_Banking_&_Account_Services_Guide.pdf",
}


class DomainResult(BaseModel):
    """The only answers the LLM is allowed to give."""

    domain: Domain = Field(description="Which NorthStar Bank guide the question is about")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the NorthStar Bank customer question into one domain:\n"
     "- credit_card: credit cards, card fees, APRs, rewards, balance transfers\n"
     "- personal_loan: personal loans, loan rates, payments, payoff\n"
     "- digital_banking: checking, savings, mobile app, transfers, overdraft, debit cards\n"
     "- all: unclear, or the question covers more than one domain"),
    ("human", "{question}"),
])


def classify_domain(question: str) -> Domain:
    """Ask the LLM which guide the question is about."""
    chain = PROMPT | get_llm().with_structured_output(DomainResult)
    return chain.invoke({"question": question}).domain


def domain_to_source(domain: Domain) -> str | None:
    """Map a domain to its PDF name: 'credit_card' -> PDF name, 'all' -> None (every guide)."""
    return GUIDES.get(domain)
