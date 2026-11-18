"""Build the search index: PDFs -> chunks -> embeddings in Chroma.

Run once (and again whenever the PDFs change):
    uv run python -m northstar_rag.ingestion.pipeline
"""

from northstar_rag.ingestion.splitter import load_all_chunks
from northstar_rag.retrieval import build_vector_store


def build_index() -> int:
    """Load all PDFs, embed the chunks, save them. Returns how many chunks are stored."""
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks from the PDFs")
    store = build_vector_store(chunks)
    # _collection is private (Chroma has no public count); re-check after Chroma upgrades
    count = store._collection.count()
    print(f"Stored {count} chunks in Chroma")
    return count


if __name__ == "__main__":
    build_index()
