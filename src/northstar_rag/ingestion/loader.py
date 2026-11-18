"""Load a PDF and turn it into clean Markdown text."""

import re
from pathlib import Path

import pymupdf4llm

# Matches the footer lines repeated on every page, so they don't end up in the chunks:
#   1) any line containing a document id like "(NSB-CC-2025-V1)"
#   2) lines like "Page 3 of 20"
# re.M makes ^ and $ match at the start/end of every line, not just the whole text.
FOOTER_RE = re.compile(r"^(?:.*\(NSB-[A-Z]+-\d{4}-V\d+\).*|\s*Page \d+ of \d+\s*)$\n?", re.M)


def load_pdf_as_markdown(pdf_path: Path) -> str:
    """Convert a PDF to Markdown (keeps headings and tables) and remove page footers."""
    md_text = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)
    return FOOTER_RE.sub("", md_text)
