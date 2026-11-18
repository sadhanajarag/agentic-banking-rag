"""Embed chunks with OpenAI and store them in a Chroma database on disk."""

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from northstar_rag.config import settings


def get_vector_store() -> Chroma:
    """Open the Chroma collection on disk (it is created if it does not exist)."""
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def make_chunk_id(chunk: Document) -> str:
    """Same chunk always gets the same id."""
    m = chunk.metadata
    text = f"{m['source']}|{m['chunk_index']}|{chunk.page_content}"
    return hashlib.md5(text.encode()).hexdigest()


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Wipe the collection, then embed and save all chunks."""
    store = get_vector_store()
    store.reset_collection()
    store.add_documents(chunks, ids=[make_chunk_id(c) for c in chunks])
    return store
