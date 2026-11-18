"""Rerank: a cross-encoder model re-scores the hybrid results and keeps the best few."""

from functools import cache

import torch
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from northstar_rag.config import settings
from northstar_rag.retrieval.hybrid import build_hybrid_retriever


class FastCrossEncoder(HuggingFaceCrossEncoder):
    """HuggingFaceCrossEncoder that scores in small batches, without a progress bar.

    LangChain's version always uses batches of 32. Every pair in a batch is padded
    to the longest one (up to 512 tokens), so big batches waste most of the CPU time.
    """

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """Return one relevance score per (question, chunk) pair."""
        scores = self.client.predict(
            text_pairs, batch_size=settings.rerank_batch_size, show_progress_bar=False
        )
        return scores.tolist()


@cache  # @cache = remember the result, so the model loads only once
def get_reranker() -> CrossEncoderReranker:
    """Load the reranker model only once, then reuse it."""
    # Applies to the whole process; only the reranker uses torch in this app
    torch.set_num_threads(settings.torch_threads)
    model = FastCrossEncoder(model_name=settings.reranker_model)
    return CrossEncoderReranker(model=model, top_n=settings.rerank_top_n)


# Cached per source (at most 4: one per guide + None for "all"). Building a retriever
# reads every chunk from Chroma and rebuilds the BM25 index, so without the cache that
# would happen on every search - up to 4 times per question with the rewrite loop.
# After re-running the ingestion pipeline, restart the app to pick up the new index.
@cache
def build_reranked_retriever(source: str | None = None) -> ContextualCompressionRetriever:
    """Hybrid search (top retrieval_k) -> reranker -> best rerank_top_n chunks.

    source = one PDF file name, or None to search every guide.
    """
    return ContextualCompressionRetriever(
        base_compressor=get_reranker(),
        base_retriever=build_hybrid_retriever(source),
    )
