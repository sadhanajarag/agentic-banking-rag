import pytest
from pydantic import ValidationError

from northstar_rag.config import Settings, settings


def test_settings_load():
    assert settings.openai_api_key != ""
    assert settings.max_rewrites == 3


def test_data_folder_has_pdfs():
    assert len(list(settings.data_dir.glob("*.pdf"))) == 3


def test_bad_chunk_overlap_rejected():
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=200)


def test_bad_weights_rejected():
    with pytest.raises(ValidationError):
        Settings(bm25_weight=0.5, vector_weight=0.6)


def test_negative_k_rejected():
    with pytest.raises(ValidationError):
        Settings(retrieval_k=-1)
