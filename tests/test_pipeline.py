from types import SimpleNamespace

from northstar_rag.ingestion import pipeline


def test_build_index_returns_count(monkeypatch):
    fake_store = SimpleNamespace(_collection=SimpleNamespace(count=lambda: 3))
    monkeypatch.setattr(pipeline, "load_all_chunks", lambda: ["a", "b", "c"])
    monkeypatch.setattr(pipeline, "build_vector_store", lambda chunks: fake_store)
    assert pipeline.build_index() == 3
