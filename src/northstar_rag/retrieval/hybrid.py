"""Hybrid retrieval: BM25 (keywords) + Chroma (meaning), combined with weights."""

from typing import Any

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from northstar_rag.config import settings
from northstar_rag.retrieval.vector_store import get_vector_store


def get_chunks_from_store(store: Chroma) -> list[Document]:
    """Read all saved chunks back from Chroma (no PDF parsing, no OpenAI call)."""
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"], strict=True)
    ]


def build_bm25_retriever(chunks: list[Document], source: str | None = None) -> BM25Retriever:
    """Keyword search over the chunks (only one PDF if source is given)."""
    if source is not None:
        chunks = [chunk for chunk in chunks if chunk.metadata.get("source") == source]
    if not chunks:
        raise ValueError(
            "No chunks found. Run the ingestion pipeline first, or check the source name."
        )
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.retrieval_k
    return bm25


def build_hybrid_retriever(source: str | None = None) -> EnsembleRetriever:
    """BM25 + vector search together. source = one PDF file name, or None for all."""
    store = get_vector_store()
    bm25 = build_bm25_retriever(get_chunks_from_store(store), source)

    # Apply the same source filter to the vector search as to BM25
    search_kwargs: dict[str, Any] = {"k": settings.retrieval_k}
    if source is not None:
        search_kwargs["filter"] = {"source": source}
    dense = store.as_retriever(search_kwargs=search_kwargs)

    # EnsembleRetriever merges both result lists with Reciprocal Rank Fusion,
    # using these weights (they must add up to 1.0 - checked in config.py)
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[settings.bm25_weight, settings.vector_weight],
    )
