import pytest

from rag.errors import DocumentError
from rag.loader import discover_pdfs, load_pdf, parse_pdf_bytes, sha256_bytes
from tests.helpers import make_pdf, sds_pages


def test_parses_text_pdf_with_metadata():
    data = make_pdf(sds_pages("Acetone", "67-64-1", "-17 C"), title="Safety Data Sheet: Acetone")
    doc = parse_pdf_bytes(data, "acetone.pdf")
    assert doc.page_count == 4
    assert doc.title == "Safety Data Sheet: Acetone"
    assert "Flash point -17 C" in doc.pages[2]
    assert doc.sha256 == sha256_bytes(data)
    assert doc.filename == "acetone.pdf"


def test_rejects_non_pdf():
    with pytest.raises(DocumentError, match="not a PDF"):
        parse_pdf_bytes(b"hello world, definitely not a pdf", "x.pdf")


def test_rejects_corrupt_pdf():
    with pytest.raises(DocumentError, match=r"corrupt|unreadable|no extractable|no pages"):
        parse_pdf_bytes(b"%PDF-1.4\n garbage garbage \n%%EOF", "broken.pdf")


def test_rejects_oversized_file():
    data = make_pdf(sds_pages("Acetone", "67-64-1", "-17 C"))
    with pytest.raises(DocumentError, match="limit"):
        parse_pdf_bytes(data, "big.pdf", max_bytes=100)


def test_rejects_too_many_pages():
    data = make_pdf([["SECTION 1: Identification", "text " * 40]] * 5)
    with pytest.raises(DocumentError, match="page limit"):
        parse_pdf_bytes(data, "long.pdf", max_pages=3)


def test_rejects_image_only_pdf():
    data = make_pdf([[], []])  # pages without any text: what a scanned PDF looks like
    with pytest.raises(DocumentError, match="scanned"):
        parse_pdf_bytes(data, "scan.pdf")


def test_discover_skips_hidden_and_partial_files(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "b.PDF").write_bytes(b"x")
    (tmp_path / "uploads" / ".b.pdf.part").write_bytes(b"x")
    (tmp_path / "uploads" / ".backup-1-c.pdf").write_bytes(b"x")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "d.pdf").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    found = [p.relative_to(tmp_path).as_posix() for p in discover_pdfs(tmp_path)]
    assert found == ["a.pdf", "uploads/b.PDF"]


def test_load_pdf_uses_relative_source(tmp_path):
    sub = tmp_path / "uploads"
    sub.mkdir()
    (sub / "x.pdf").write_bytes(make_pdf(sds_pages("Ethanol", "64-17-5", "12 C")))
    doc = load_pdf(sub / "x.pdf", tmp_path)
    assert doc.source == "uploads/x.pdf"
