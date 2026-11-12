import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Put your PDF files in a folder called "pdfs" next to this file.
PDF_DIR = Path(__file__).parent / "pdfs"

CHUNK_SIZE = 700   # max characters per chunk (the embedding model reads ~256 tokens)
OVERLAP = 100      # characters repeated between chunks so sentences aren't cut off
MIN_CHUNK = 50     # ignore tiny fragments (page numbers, headers)


def _clean(text: str) -> str:
    """Collapse line breaks and repeated spaces into single spaces."""
    return " ".join(text.split())


def _split(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """Yield overlapping chunks, breaking at spaces where possible."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        yield text[start:end].strip()
        if end >= len(text):
            break
        start = end - overlap


def load_pdf_documents():
    """Return a list of {"id", "title", "content"} dicts, one per text chunk."""
    docs = []

    if not PDF_DIR.exists():
        logger.warning("No '%s' folder found, skipping PDFs.", PDF_DIR.name)
        return docs

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", PDF_DIR.name)
        return docs

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:
            logger.error("Could not open %s: %s", pdf_path.name, e)
            continue

        readable_name = pdf_path.stem.replace("_", " ").replace("-", " ")
        pages_without_text = 0
        chunks_from_file = 0

        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = _clean(page.extract_text() or "")
            except Exception:
                text = ""

            if not text:
                pages_without_text += 1
                continue

            for n, chunk in enumerate(_split(text), start=1):
                if len(chunk) < MIN_CHUNK:
                    continue
                docs.append(
                    {
                        "id": f"{pdf_path.stem}-p{page_no}-c{n}",
                        "title": f"{readable_name} (page {page_no})",
                        "source": readable_name,
                        "content": chunk,
                    }
                )
                chunks_from_file += 1

        logger.info("%s: %d chunks", pdf_path.name, chunks_from_file)
        if pages_without_text:
            logger.warning(
                "%s: %d page(s) had no extractable text (scanned images need OCR).",
                pdf_path.name,
                pages_without_text,
            )

    return docs
