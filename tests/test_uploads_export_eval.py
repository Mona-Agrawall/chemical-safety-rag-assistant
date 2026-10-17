from datetime import datetime

import pytest

from rag.citations import Source
from rag.evaluation import (
    EvalQuestion,
    Expected,
    RetrievedRef,
    facts_match,
    is_refusal,
    load_questions,
    score_retrieval,
    summarise,
)
from rag.export import DISCLAIMER, ExportMessage, conversation_to_markdown
from rag.uploads import UploadStatus, sanitize_filename, stage_uploads
from tests.helpers import make_pdf, sds_pages

# ── uploads ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Acetone SDS.pdf", "Acetone_SDS.pdf"),
        ("../../etc/passwd", "passwd.pdf"),
        ("C:\\Users\\x\\evil name.PDF", "evil_name.pdf"),
        (".hidden.pdf", "hidden.pdf"),
        ("???.pdf", "document.pdf"),
    ],
)
def test_sanitize_filename(name, expected):
    assert sanitize_filename(name) == expected


def pdf(name="Benzene", cas="71-43-2", flash="-11 C") -> bytes:
    return make_pdf(sds_pages(name, cas, flash))


def test_stage_accepts_rejects_and_dedupes(kb, corpus):
    acetone_bytes = (corpus.docs_dir / "acetone_sds.pdf").read_bytes()
    staged = stage_uploads(corpus, [
        ("benzene.pdf", pdf()),
        ("notes.pdf", b"plain text pretending to be a pdf"),
        ("again.pdf", acetone_bytes),
        ("benzene-copy.pdf", pdf()),
    ], kb.manifest)
    status = {o.filename: o.status for o in staged.outcomes}
    assert status == {
        "benzene.pdf": UploadStatus.ADDED,
        "notes.pdf": UploadStatus.REJECTED,
        "again.pdf": UploadStatus.DUPLICATE,
        "benzene-copy.pdf": UploadStatus.DUPLICATE,
    }
    assert (corpus.uploads_dir / "benzene.pdf").exists()
    assert not list(corpus.uploads_dir.glob(".*.part"))


def test_stage_rejects_oversized(kb, corpus):
    small = corpus.with_overrides(max_upload_mb=1)
    staged = stage_uploads(small, [("big.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024))], kb.manifest)
    assert staged.outcomes[0].status is UploadStatus.REJECTED and "limit" in staged.outcomes[0].message


def test_replace_then_rollback_restores_previous_file(kb, corpus):
    first = stage_uploads(corpus, [("b.pdf", pdf(flash="-11 C"))], kb.manifest)
    first.commit()
    original = (corpus.uploads_dir / "b.pdf").read_bytes()
    second = stage_uploads(corpus, [("b.pdf", pdf(flash="-12 C"))], kb.manifest)
    assert second.outcomes[0].status is UploadStatus.REPLACED
    assert (corpus.uploads_dir / "b.pdf").read_bytes() != original
    second.rollback()
    assert (corpus.uploads_dir / "b.pdf").read_bytes() == original
    assert not list(corpus.uploads_dir.glob(".backup-*"))


def test_commit_discards_backup_and_rollback_removes_new_file(kb, corpus):
    stage_uploads(corpus, [("b.pdf", pdf(flash="-11 C"))], kb.manifest).commit()
    replaced = stage_uploads(corpus, [("b.pdf", pdf(flash="-13 C"))], kb.manifest)
    replaced.commit()
    assert not list(corpus.uploads_dir.glob(".backup-*"))
    new = stage_uploads(corpus, [("c.pdf", pdf("Toluene", "108-88-3"))], kb.manifest)
    new.rollback()
    assert not (corpus.uploads_dir / "c.pdf").exists()


def test_uploaded_document_becomes_searchable(kb, corpus, embedder):
    from rag.indexing import load_knowledge_base, sync_knowledge_base
    from rag.retrieval import Retriever

    staged = stage_uploads(corpus, [("benzene.pdf", pdf())], kb.manifest)
    report = sync_knowledge_base(corpus, embedder)
    staged.commit()
    assert report.added == ["uploads/benzene.pdf"]
    new_kb = load_knowledge_base(corpus)
    res = Retriever(new_kb, embedder, corpus).retrieve("flash point of benzene")
    assert res.chunks and all(r.chunk.source == "uploads/benzene.pdf" for r in res.chunks)


def test_partial_index_failure_rolls_back_only_the_failed_upload(kb, corpus, embedder, monkeypatch):
    """Regression: one failed upload used to roll back every upload, including ones already indexed."""
    from rag import indexing
    from rag.errors import DocumentError
    from rag.indexing import KBState, assess_status, load_knowledge_base, sync_knowledge_base

    real = indexing.process_document

    def flaky(doc, settings):
        if doc.filename == "toluene.pdf":
            raise DocumentError("toluene.pdf: no usable text after processing.")
        return real(doc, settings)

    monkeypatch.setattr(indexing, "process_document", flaky)
    staged = stage_uploads(corpus, [("benzene.pdf", pdf()), ("toluene.pdf", pdf("Toluene", "108-88-3"))],
                           kb.manifest)
    report = sync_knowledge_base(corpus, embedder)
    kept, rolled_back = staged.settle(corpus.docs_dir, dict(report.failed))

    assert [o.filename for o in kept] == ["benzene.pdf"]
    assert [o.filename for o in rolled_back] == ["toluene.pdf"] and "no usable text" in rolled_back[0].message
    assert (corpus.uploads_dir / "benzene.pdf").exists() and not (corpus.uploads_dir / "toluene.pdf").exists()
    new_kb = load_knowledge_base(corpus)
    assert "benzene" in {i.key for i in new_kb.identities}
    assert assess_status(corpus, new_kb, None).state is KBState.READY


def test_failed_replacement_restores_previous_upload_and_index(kb, corpus, embedder, monkeypatch):
    from rag import indexing
    from rag.errors import DocumentError
    from rag.indexing import KBState, assess_status, load_knowledge_base, sync_knowledge_base

    stage_uploads(corpus, [("b.pdf", pdf(flash="-11 C"))], kb.manifest).commit()
    sync_knowledge_base(corpus, embedder)
    original = (corpus.uploads_dir / "b.pdf").read_bytes()

    def broken(doc, settings):
        raise DocumentError("b.pdf: no usable text after processing.")

    monkeypatch.setattr(indexing, "process_document", broken)
    replaced = stage_uploads(corpus, [("b.pdf", pdf(flash="-12 C"))], load_knowledge_base(corpus).manifest)
    report = sync_knowledge_base(corpus, embedder)
    _, rolled_back = replaced.settle(corpus.docs_dir, dict(report.failed))
    monkeypatch.undo()
    assert rolled_back and (corpus.uploads_dir / "b.pdf").read_bytes() == original
    sync_knowledge_base(corpus, embedder)  # what the app does after a rollback
    new_kb = load_knowledge_base(corpus)
    assert any("-11 C" in c.text for c in new_kb.chunks if c.source == "uploads/b.pdf")
    assert assess_status(corpus, new_kb, None).state is KBState.READY


# ── export ────────────────────────────────────────────────────────────────


def test_markdown_export_includes_questions_answers_and_citations():
    s = Source("S1", "acetone_sds.pdf", "Acetone", 9, "Physical and chemical properties", 8, 8, "Flash point -17 C")
    md = conversation_to_markdown(
        [ExportMessage("user", "Flash point of acetone?"),
         ExportMessage("assistant", "It is -17 C [1].", [s], ["Check the SDS."])],
        model="mistral", index_version="v1", generated_at=datetime(2026, 1, 2, 3, 4),
    )
    assert "## Question 1" in md and "Flash point of acetone?" in md and "It is -17 C [1]." in md
    assert "1. acetone_sds.pdf, p. 8, Acetone, Section 9: Physical and chemical properties" in md
    assert DISCLAIMER in md and "2026-01-02 03:04" in md and "**Note:** Check the SDS." in md


# ── evaluation metrics ────────────────────────────────────────────────────


def test_facts_match_normalises_decimal_commas_and_minus_signs():
    assert facts_match("Explosion limits 2,6 vol% and 12.8 vol%", (("2.6",), ("12.8",)))
    assert facts_match("Flash point \u221217 \u00b0C", (("-17",),))
    assert not facts_match("Flash point 17 C", (("-17",), ("°c",)))


def test_refusal_detection():
    assert is_refusal("The provided Safety Data Sheets do not contain this information.")
    assert not is_refusal("The flash point is -17 °C [S1].")


def test_score_retrieval_multi_source_and_contamination():
    q = EvalQuestion("q", "multi", "?", ("acetone", "methanol"), True,
                     (Expected("acetone_sds.pdf", (8,)), Expected("methanol_sds.pdf", (6,))), ())
    refs = [RetrievedRef("acetone_sds.pdf", 8, 8), RetrievedRef("ethanol_sds.pdf", 6, 6),
            RetrievedRef("uploads/methanol_sds.pdf", 5, 6)]
    chem = {"acetone": "acetone_sds.pdf", "methanol": "methanol_sds.pdf", "ethanol": "ethanol_sds.pdf"}
    r = score_retrieval(q, refs, (1, 3), chem)
    assert r.hit_at == {1: False, 3: True}
    assert r.reciprocal_rank == pytest.approx((1 + 1 / 3) / 2)
    assert r.contamination == pytest.approx(1 / 3)
    s = summarise([r], (1, 3))
    assert s["hit@3"] == 1.0 and s["questions_with_any_contamination"] == 1


def test_bundled_eval_set_is_well_formed():
    from pathlib import Path

    qs = load_questions(Path(__file__).resolve().parent.parent / "eval" / "questions.json")
    assert 30 <= len(qs) <= 60
    assert len({q.id for q in qs}) == len(qs)
    for q in qs:
        assert q.answerable or not q.facts
        for e in q.expected:
            assert e.pages and all(p >= 1 for p in e.pages)
