import pytest
from langchain_core.documents import Document

from northstar_rag.retrieval import build_bm25_retriever

CHUNKS = [
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


def test_bm25_finds_keyword():
    results = build_bm25_retriever(CHUNKS).invoke("foreign transaction fee")
    assert results[0].metadata["source"] == "card.pdf"


def test_bm25_source_filter():
    results = build_bm25_retriever(CHUNKS, source="loan.pdf").invoke("grace period")
    assert all(r.metadata["source"] == "loan.pdf" for r in results)


def test_bm25_empty_raises():
    with pytest.raises(ValueError):
        build_bm25_retriever(CHUNKS, source="missing.pdf")
