"""Relevance grader: keep only the chunks that actually help answer the question."""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from northstar_rag.llm.client import get_llm


class GradeResult(BaseModel):
    """The LLM's yes/no answer for one chunk."""

    relevant: bool = Field(description="True if the chunk helps answer the question")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether a chunk from a NorthStar Bank guide is relevant to a customer question.\n"
     "Mark it relevant if it contains information that helps answer the question, even partly.\n"
     "Otherwise mark it not relevant."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])


def grade_documents(question: str, docs: list[Document]) -> list[Document]:
    """Return only the docs the LLM marks as relevant.

    All chunks are graded in parallel with .batch() (one LLM call per chunk).
    """
    if not docs:
        return []
    chain = PROMPT | get_llm().with_structured_output(GradeResult)
    grades = chain.batch([{"question": question, "chunk": doc.page_content} for doc in docs])
    # strict=True: raise an error if we ever get a different number of grades than docs
    return [doc for doc, grade in zip(docs, grades, strict=True) if grade.relevant]
