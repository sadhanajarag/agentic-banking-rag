"""Generator: write the final answer from the relevant chunks, with [1], [2] citations."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from northstar_rag.llm.client import get_llm

FALLBACK_ANSWER = (
    "I couldn't find that in NorthStar Bank's guides. "
    "Please contact NorthStar Bank customer service for help."
)

# Note: only the FALLBACK_ANSWER line is an f-string. "{context}" must stay a
# plain string so the prompt template can fill it in later.
SYSTEM_PROMPT = (
    "You are NorthStar Bank's customer-support assistant.\n"
    "Answer ONLY from the numbered context below. Rules:\n"
    f'- If the answer is not in the context, say: "{FALLBACK_ANSWER}" Do not guess.\n'
    "- Quote exact figures (fees, APRs, limits, phone numbers, timeframes) as written.\n"
    "- Context may contain Markdown tables: "
    "read the correct row AND column (e.g. the specific card).\n"
    "- If the question is ambiguous between products, "
    "give the answer for each relevant product.\n"
    "- Cite sources inline as [1], [2] matching the context numbers.\n"
    "- Be concise: a direct answer first, then brief supporting details.\n\n"
    "Context:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_context(docs: list[Document]) -> str:
    """Number the chunks [1], [2] ... with source and section, so the LLM can cite them."""
    # Each block starts like: "[1] Source: card.pdf | Section: 5. Fees > Late fees | contains TABLE"
    blocks = []
    for number, doc in enumerate(docs, start=1):
        meta = doc.metadata
        heading = " > ".join(h for h in (meta.get("section"), meta.get("subsection")) if h)
        table_note = " | contains TABLE" if meta.get("has_table") else ""
        blocks.append(
            f"[{number}] Source: {meta.get('source')} | Section: {heading or 'n/a'}{table_note}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, docs: list[Document]) -> str:
    """Answer the question from the docs. No docs -> fallback message (no LLM call)."""
    if not docs:
        return FALLBACK_ANSWER
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"question": question, "context": format_context(docs)})
