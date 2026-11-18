from langchain_core.documents import Document

from northstar_rag.config import settings
from northstar_rag.retrieval import get_reranker

DOCS = [
    Document(
        page_content="Foreign transaction fee is $0 on Travel Horizon",
        metadata={"source": "card.pdf"},
    ),
    Document(page_content="Personal loan grace period is 10 days", metadata={"source": "loan.pdf"}),
    Document(
        page_content="Overdraft grace until 11 PM next business day",
        metadata={"source": "digital.pdf"},
    ),
]


def test_reranker_puts_best_first():
    results = get_reranker().compress_documents(DOCS, "What is the foreign transaction fee?")
    assert results[0].metadata["source"] == "card.pdf"


def test_reranker_keeps_at_most_top_n():
    results = get_reranker().compress_documents(DOCS * 3, "grace period")  # 9 docs in
    assert len(results) == settings.rerank_top_n                          # 4 docs out
