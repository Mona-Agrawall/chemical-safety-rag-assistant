"""PDF parsing and validation.

The same code path is used by the CLI ingester and by in-app uploads, so a
document is processed identically no matter how it enters the knowledge base.
"""

from __future__ import annotations

import hashlib
import io
import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.errors import DocumentError

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
# Below this many extractable characters per page on average we treat the
# file as scanned / image-only. Real SDS pages carry 1,000+ characters.
MIN_CHARS_PER_PAGE = 80


@dataclass
class PdfDocument:
    """Text and provenance of one PDF, pages are 1-based in ``page_numbers``."""

    source: str  # path relative to the documents directory, POSIX separators
    sha256: str
    size_bytes: int
    pages: list[str]
    title: str = ""
    page_labels: list[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return Path(self.source).name

    @property
    def page_count(self) -> int:
        return len(self.pages)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_page_text(text: str) -> str:
    """Normalise unicode and strip characters that confuse tokenisers."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("−", "-").replace("\x00", "")
    # Drop other control characters but keep newlines/tabs.
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def parse_pdf_bytes(
    data: bytes,
    source: str,
    *,
    max_bytes: int | None = None,
    max_pages: int | None = None,
) -> PdfDocument:
    """Validate and extract text from PDF bytes.

    Raises :class:`DocumentError` with a user-presentable message for files
    that are not PDFs, are encrypted, corrupt, too large, or have no
    extractable text (scanned images).
    """
    name = Path(source).name
    if max_bytes is not None and len(data) > max_bytes:
        raise DocumentError(
            f"{name}: file is {len(data) / 1e6:.1f} MB, above the {max_bytes / 1e6:.0f} MB limit."
        )
    if not data.lstrip()[:5].startswith(PDF_MAGIC):
        raise DocumentError(f"{name}: not a PDF file (missing %PDF header).")

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Many "encrypted" SDS files use an empty user password.
            try:
                if not reader.decrypt(""):
                    raise DocumentError(f"{name}: PDF is password-protected.")
            except DocumentError:
                raise
            except Exception as exc:
                raise DocumentError(f"{name}: PDF is encrypted and cannot be read.") from exc
        n_pages = len(reader.pages)
    except DocumentError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise DocumentError(f"{name}: PDF is corrupt or unreadable ({exc.__class__.__name__}).") from exc

    if n_pages == 0:
        raise DocumentError(f"{name}: PDF has no pages.")
    if max_pages is not None and n_pages > max_pages:
        raise DocumentError(f"{name}: {n_pages} pages exceeds the {max_pages}-page limit.")

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(clean_page_text(page.extract_text() or ""))
        except Exception as exc:  # pypdf raises a wide range of errors on bad pages
            log.warning("Text extraction failed on %s page %d: %s", name, i + 1, exc)
            pages.append("")

    total_chars = sum(len(p.strip()) for p in pages)
    if total_chars == 0:
        raise DocumentError(
            f"{name}: no extractable text. It is probably a scanned image; "
            "OCR is not supported, please upload a text-based PDF."
        )
    if total_chars / n_pages < MIN_CHARS_PER_PAGE:
        raise DocumentError(
            f"{name}: very little extractable text ({total_chars} characters over {n_pages} pages). "
            "It is probably scanned; OCR is not supported."
        )

    title = ""
    try:
        meta = reader.metadata
        if meta and meta.title:
            title = clean_page_text(str(meta.title)).strip()
    except Exception:  # metadata is optional and often malformed
        title = ""
    try:
        labels = list(reader.page_labels)
    except Exception:
        labels = []

    return PdfDocument(
        source=source,
        sha256=sha256_bytes(data),
        size_bytes=len(data),
        pages=pages,
        title=title,
        page_labels=labels if len(labels) == n_pages else [],
    )


def load_pdf(path: Path, docs_dir: Path, **limits: int | None) -> PdfDocument:
    """Read and parse a PDF on disk; ``source`` is relative to ``docs_dir``."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DocumentError(f"{path.name}: cannot read file ({exc}).") from exc
    return parse_pdf_bytes(data, relative_source(path, docs_dir), **limits)


def relative_source(path: Path, docs_dir: Path) -> str:
    try:
        return path.resolve().relative_to(docs_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def discover_pdfs(docs_dir: Path) -> list[Path]:
    """All PDFs under ``docs_dir`` (recursive), sorted for determinism.

    Hidden directories and partially written uploads (``*.part``) are skipped.
    """
    if not docs_dir.exists():
        return []
    found = []
    for p in docs_dir.rglob("*"):
        rel = p.relative_to(docs_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if p.is_file() and p.suffix.lower() == ".pdf":
            found.append(p)
    return sorted(found, key=lambda p: p.relative_to(docs_dir).as_posix().lower())
