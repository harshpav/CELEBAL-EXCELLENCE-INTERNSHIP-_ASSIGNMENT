"""
Document Ingestion Module
=========================
Loads PDF and plain-text documents, extracts raw text, and returns a list of
Document objects (text + metadata) ready for chunking.

Supported formats
-----------------
- .pdf  — via pypdf
- .txt  — plain UTF-8 / latin-1
- .md   — treated as plain text

Design decisions
----------------
- Each *page* of a PDF becomes its own Document so that page-level metadata
  (source, page number) travels with every chunk downstream.
- Text is normalised: excessive whitespace is collapsed, BOM characters removed.
- Preprocessing is kept separate from loading so tests can call it in isolation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from loguru import logger


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Minimal document container used throughout the pipeline."""
    text: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Document text must not be empty.")


# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------

_MULTI_SPACE_RE = re.compile(r" {2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def preprocess_text(text: str) -> str:
    """
    Clean raw extracted text.

    Steps:
    1. Remove BOM / non-printable control characters.
    2. Normalise unicode to NFC.
    3. Collapse multiple spaces to one.
    4. Limit consecutive blank lines to two.
    5. Strip leading/trailing whitespace.
    """
    # Remove BOM
    text = text.lstrip("\ufeff")
    # Remove control characters (keep \t, \n, \r)
    text = _CONTROL_CHARS_RE.sub("", text)
    # Unicode normalisation
    text = unicodedata.normalize("NFC", text)
    # Collapse spaces
    text = _MULTI_SPACE_RE.sub(" ", text)
    # Limit blank lines
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> list[Document]:
    """
    Load a PDF file page by page using pypdf.
    Returns one Document per page (skips blank pages).
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF loading: pip install pypdf") from exc

    reader = PdfReader(str(path))
    docs: list[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = preprocess_text(raw)
        if not cleaned:
            logger.debug("Skipping blank page {}/{} in {}", page_num, len(reader.pages), path.name)
            continue
        docs.append(Document(
            text=cleaned,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": "pdf",
                "page": page_num,
                "total_pages": len(reader.pages),
            },
        ))

    logger.info("Loaded {} page(s) from PDF: {}", len(docs), path.name)
    return docs


def _load_text(path: Path) -> list[Document]:
    """
    Load a plain-text or Markdown file.
    Returns a single Document for the whole file.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode {path} as UTF-8 or latin-1.")

    cleaned = preprocess_text(raw)
    if not cleaned:
        raise ValueError(f"File is empty after preprocessing: {path}")

    doc = Document(
        text=cleaned,
        metadata={
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix.lstrip(".") or "txt",
            "page": 1,
        },
    )
    logger.info("Loaded text file: {} ({} chars)", path.name, len(cleaned))
    return [doc]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADER_MAP: dict[str, callable] = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_text,
    ".markdown": _load_text,
}


def load_document(path: Union[str, Path]) -> list[Document]:
    """
    Load a single document from *path*.

    Parameters
    ----------
    path : str | Path
        Path to a PDF, TXT, or MD file.

    Returns
    -------
    list[Document]
        One or more Document objects (multiple for PDFs with many pages).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file type is unsupported or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    suffix = path.suffix.lower()
    loader = _LOADER_MAP.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {list(_LOADER_MAP.keys())}"
        )

    return loader(path)


def load_directory(
    directory: Union[str, Path],
    recursive: bool = True,
    extensions: tuple[str, ...] = (".pdf", ".txt", ".md"),
) -> list[Document]:
    """
    Load all supported documents from a directory.

    Parameters
    ----------
    directory : str | Path
        Root directory to scan.
    recursive : bool
        Whether to recurse into sub-directories.
    extensions : tuple[str, ...]
        File extensions to include (lowercase, with leading dot).

    Returns
    -------
    list[Document]
        All successfully loaded documents.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    all_docs: list[Document] = []
    errors: list[str] = []

    for path in sorted(directory.glob(pattern)):
        if path.suffix.lower() not in extensions:
            continue
        try:
            docs = load_document(path)
            all_docs.extend(docs)
        except Exception as exc:
            logger.warning("Failed to load {}: {}", path, exc)
            errors.append(str(path))

    logger.info(
        "Directory load complete: {} document(s) from {} file(s). {} error(s).",
        len(all_docs),
        len(all_docs),
        len(errors),
    )
    return all_docs
