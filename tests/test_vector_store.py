from langchain_core.documents import Document

from northstar_rag.retrieval import get_vector_store, make_chunk_id


def make_chunk(text, index=0):
    return Document(page_content=text, metadata={"source": "a.pdf", "chunk_index": index})


def test_same_chunk_same_id():
    assert make_chunk_id(make_chunk("hello")) == make_chunk_id(make_chunk("hello"))


def test_different_chunk_different_id():
    assert make_chunk_id(make_chunk("hello", 0)) != make_chunk_id(make_chunk("hello", 1))


def test_store_opens():
    store = get_vector_store()
    assert store._collection.name == "northstar_docs"
