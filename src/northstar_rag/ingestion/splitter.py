"""Split Markdown text into chunks (LangChain Documents)."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from northstar_rag.config import settings
from northstar_rag.ingestion.loader import load_pdf_as_markdown

# Step 1: split at Markdown headings and remember them as metadata
# (title / section / subsection). strip_headers=False keeps the heading in the text.
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")],
    strip_headers=False,
)

# Step 2: cut sections that are still too long, preferring to cut at
# headings, then paragraphs, then lines, then words.
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)


def split_pdf(pdf_path: Path) -> list[Document]:
    """Turn one PDF into chunks with source, chunk_index and has_table metadata."""
    md_text = load_pdf_as_markdown(pdf_path)
    sections = header_splitter.split_text(md_text)
    chunks = char_splitter.split_documents(sections)
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            source=pdf_path.name,
            chunk_index=index,
            has_table="|---|" in chunk.page_content,  # Markdown table separator row
        )
    return chunks


def load_all_chunks(data_dir: Path = settings.data_dir) -> list[Document]:
    """Split every PDF in data_dir and return all chunks in one list."""
    all_chunks: list[Document] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        all_chunks.extend(split_pdf(pdf_path))
    return all_chunks
