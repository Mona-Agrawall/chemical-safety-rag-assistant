"""Regression tests on the four bundled SDS PDFs in data/ (real layouts, real model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chemicals import extract_identity
from rag.chunking import chunk_document
from rag.config import BASE_DIR, Settings
from rag.loader import discover_pdfs, load_pdf

DATA = BASE_DIR / "data"
EXPECTED = {
    "acetone_sds.pdf": ("acetone", "67-64-1"),
    "benzene_sds.pdf": ("benzene", "71-43-2"),
    "ethanol_sds.pdf": ("ethanol", "64-17-5"),
    "methanol_sds.pdf": ("methanol", "67-56-1"),
}
pytestmark = pytest.mark.skipif(
    not all((DATA / f).exists() for f in EXPECTED), reason="bundled SDS PDFs not present"
)


@pytest.fixture(scope="module")
def docs():
    return {p.name: load_pdf(p, DATA) for p in discover_pdfs(DATA) if p.name in EXPECTED}


def test_identity_of_every_bundled_sds(docs):
    for name, (key, cas) in EXPECTED.items():
        d = docs[name]
        ident = extract_identity(d.pages, d.title, d.filename)
        assert (ident.key, ident.cas) == (key, cas), name


def test_all_sixteen_sections_detected(docs):
    for name, d in docs.items():
        ident = extract_identity(d.pages, d.title, d.filename)
        chunks = chunk_document(d.pages, ident, d.source, d.sha256, 1000, 150)
        assert {c.section for c in chunks} >= set(range(1, 17)), name


@pytest.mark.parametrize(
    ("name", "needle", "page", "section"),
    [
        ("acetone_sds.pdf", "Flash point -17", 8, 9),
        ("benzene_sds.pdf", "Closed cup: -11", 6, 9),
        ("ethanol_sds.pdf", "Flash point 12", 6, 9),
        ("methanol_sds.pdf", "Flash point 9,7", 6, 9),
        ("acetone_sds.pdf", "Butyl caoutchouc", 7, 8),
        ("methanol_sds.pdf", "IF SWALLOWED", 2, 2),
    ],
)
def test_chunk_page_numbers_match_the_pdf(docs, name, needle, page, section):
    d = docs[name]
    ident = extract_identity(d.pages, d.title, d.filename)
    chunks = [c for c in chunk_document(d.pages, ident, d.source, d.sha256, 1000, 150) if needle in c.text]
    assert chunks, f"{needle!r} not found in {name}"
    assert any(c.page_start <= page <= c.page_end and c.section == section for c in chunks)


def _model_cached(model: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _model_cached("sentence-transformers/all-MiniLM-L6-v2"),
                    reason="embedding model not cached locally")
def test_retrieval_quality_regression(tmp_path: Path):
    """Guards the measured retrieval quality on the bundled evaluation set."""
    from rag.embeddings import Embedder
    from rag.evaluation import RetrievedRef, load_questions, score_retrieval, summarise
    from rag.indexing import load_knowledge_base, sync_knowledge_base
    from rag.retrieval import Retriever

    s = Settings(docs_dir=DATA, index_dir=tmp_path / "index")
    embedder = Embedder(s.embedding_model)
    sync_knowledge_base(s, embedder)
    kb = load_knowledge_base(s)
    retriever = Retriever(kb, embedder, s)
    chem = {d.chemical.key: d.filename for d in kb.manifest.documents}
    results = []
    for q in load_questions(BASE_DIR / "eval" / "questions.json"):
        res = retriever.retrieve(q.question, top_k=5)
        refs = [RetrievedRef(r.chunk.source, r.chunk.page_start, r.chunk.page_end) for r in res.chunks]
        results.append(score_retrieval(q, refs, (1, 5), chem))
    summary = summarise(results, (1, 5))
    assert summary["hit@5"] >= 0.95, summary
    assert summary["contamination@k"] == 0.0, summary
