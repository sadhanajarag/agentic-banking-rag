from langchain_core.documents import Document

from northstar_rag.llm import FALLBACK_ANSWER, format_context, generate_answer

DOCS = [
    Document(
        page_content="Annual fee is $95",
        metadata={"source": "card.pdf", "section": "5. Fees", "has_table": True},
    ),
    Document(page_content="Grace period is 10 days", metadata={"source": "loan.pdf"}),
]


def test_context_is_numbered():
    context = format_context(DOCS)
    assert context.startswith("[1] Source: card.pdf | Section: 5. Fees | contains TABLE")
    assert "[2] Source: loan.pdf | Section: n/a" in context


def test_no_docs_gives_fallback():
    assert generate_answer("any question", []) == FALLBACK_ANSWER


def test_prompt_has_context_and_question():
    from northstar_rag.llm.generator import PROMPT
    assert set(PROMPT.input_variables) == {"context", "question"}
