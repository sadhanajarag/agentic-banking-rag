import pytest

from northstar_rag.config import settings
from northstar_rag.ingestion import load_all_chunks, load_pdf_as_markdown


# Load the PDFs once and share the result with all tests (PDF parsing is slow)
@pytest.fixture(scope="module")
def chunks():
    return load_all_chunks()


def test_footer_removed():
    pdf_path = sorted(settings.data_dir.glob("*.pdf"))[0]
    md_text = load_pdf_as_markdown(pdf_path)
    assert "(NSB-" not in md_text  # footer form; the cover's "Document ID: NSB-..." stays
    assert "Page 1 of" not in md_text


def test_chunks_have_metadata(chunks):
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "has_table" in chunk.metadata


def test_chunk_size_respected(chunks):
    for chunk in chunks:
        assert len(chunk.page_content) <= settings.chunk_size


def test_all_three_pdfs_loaded(chunks):
    sources = {chunk.metadata["source"] for chunk in chunks}
    assert len(sources) == 3
    assert len(chunks) > 0
