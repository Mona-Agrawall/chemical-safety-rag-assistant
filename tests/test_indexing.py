import json
import threading

import pytest

from rag import indexing
from rag.config import INDEX_FORMAT_VERSION, PIPELINE_VERSION
from rag.errors import IndexBusyError, IndexCorruptError, IndexIncompatibleError, IndexMissingError
from rag.indexing import (
    KBState,
    assess_status,
    diff_sources,
    index_write_lock,
    load_knowledge_base,
    read_current_version,
    sync_knowledge_base,
)
from tests.conftest import write_sds


def test_missing_index_raises(settings):
    with pytest.raises(IndexMissingError):
        load_knowledge_base(settings)


def test_build_and_load(corpus, embedder):
    report = sync_knowledge_base(corpus, embedder)
    assert report.changed and sorted(report.added) == ["acetone_sds.pdf", "ethanol_sds.pdf", "methanol_sds.pdf"]
    kb = load_knowledge_base(corpus)
    assert kb.version == read_current_version(corpus.index_dir) == report.version
    assert kb.index.ntotal == len(kb.chunks) == kb.manifest.chunk_count > 0
    assert [i.key for i in kb.identities] == ["acetone", "ethanol", "methanol"]


def test_manifest_records_provenance(kb, corpus):
    m = kb.manifest
    assert m.format_version == INDEX_FORMAT_VERSION and m.pipeline_version == PIPELINE_VERSION
    assert m.embedding["model"] == "test/fake-embedder" and m.embedding["dimension"] == 256
    assert m.chunking == {"strategy": PIPELINE_VERSION, "chunk_size": 600, "chunk_overlap": 80}
    doc = next(d for d in m.documents if d.chemical.key == "methanol")
    assert doc.chemical.cas == "67-56-1" and doc.pages == 4 and len(doc.sha256) == 64
    assert doc.sections == list(range(1, 17))
    on_disk = json.loads((corpus.index_dir / "versions" / kb.version / "manifest.json").read_text())
    assert on_disk["documents"][0]["sha256"]


def test_unchanged_sync_is_a_no_op(kb, corpus, embedder):
    before = embedder.documents_embedded
    report = sync_knowledge_base(corpus, embedder)
    assert not report.changed and report.version == kb.version
    assert embedder.documents_embedded == before


def test_modified_document_is_reembedded_and_old_chunks_dropped(kb, corpus, embedder):
    write_sds(corpus, "ethanol_sds.pdf", "Ethanol", "64-17-5", "13 C")  # edited flash point
    before = embedder.documents_embedded
    report = sync_knowledge_base(corpus, embedder)
    assert report.updated == ["ethanol_sds.pdf"] and sorted(report.unchanged) == ["acetone_sds.pdf", "methanol_sds.pdf"]
    new_kb = load_knowledge_base(corpus)
    eth = [c for c in new_kb.chunks if c.chemical_key == "ethanol"]
    assert embedder.documents_embedded - before == len(eth)  # only ethanol re-embedded
    text = " ".join(c.text for c in eth)
    assert "13 C" in text and "Flash point 12 C" not in text


def test_removed_document_disappears(kb, corpus, embedder):
    (corpus.docs_dir / "acetone_sds.pdf").unlink()
    report = sync_knowledge_base(corpus, embedder)
    assert report.removed == ["acetone_sds.pdf"]
    new_kb = load_knowledge_base(corpus)
    assert all(c.chemical_key != "acetone" for c in new_kb.chunks)
    assert [i.key for i in new_kb.identities] == ["ethanol", "methanol"]


def test_renamed_document_keeps_vectors(kb, corpus, embedder):
    (corpus.docs_dir / "methanol_sds.pdf").rename(corpus.docs_dir / "MeOH.pdf")
    before = embedder.documents_embedded
    report = sync_knowledge_base(corpus, embedder)
    assert report.renamed == [("methanol_sds.pdf", "MeOH.pdf")]
    assert embedder.documents_embedded == before
    new_kb = load_knowledge_base(corpus)
    assert {c.source for c in new_kb.chunks if c.chemical_key == "methanol"} == {"MeOH.pdf"}


def test_duplicate_content_indexed_once(kb, corpus, embedder):
    (corpus.docs_dir / "a_copy.pdf").write_bytes((corpus.docs_dir / "acetone_sds.pdf").read_bytes())
    report = sync_knowledge_base(corpus, embedder)
    assert report.duplicates == [("a_copy.pdf", "acetone_sds.pdf")] and not report.failed
    new_kb = load_knowledge_base(corpus)
    assert len(new_kb.manifest.documents) == 3
    assert {d.source for d in new_kb.manifest.documents} >= {"acetone_sds.pdf"}


def test_duplicate_copy_does_not_leave_index_permanently_stale(kb, corpus, embedder):
    """Regression: a byte-identical copy was reported as "1 new" on every status check."""
    (corpus.docs_dir / "a_copy.pdf").write_bytes((corpus.docs_dir / "acetone_sds.pdf").read_bytes())
    sync_knowledge_base(corpus, embedder)
    status = assess_status(corpus, load_knowledge_base(corpus), None)
    assert status.state is KBState.READY, status.message
    assert status.diff is not None and status.diff.duplicates == [("a_copy.pdf", "acetone_sds.pdf")]
    again = sync_knowledge_base(corpus, embedder)
    assert not again.changed and again.duplicates == [("a_copy.pdf", "acetone_sds.pdf")]


def test_corrupt_pdf_is_skipped_and_reported(kb, corpus, embedder):
    (corpus.docs_dir / "broken.pdf").write_bytes(b"%PDF-1.4 nonsense")
    report = sync_knowledge_base(corpus, embedder)
    assert [src for src, _ in report.failed] == ["broken.pdf"]
    assert not report.changed  # nothing else changed: last good index kept
    assert load_knowledge_base(corpus).version == kb.version


def test_incompatible_embedding_model_is_refused(kb, corpus):
    other = corpus.with_overrides(embedding_model="another/model")
    with pytest.raises(IndexIncompatibleError, match="embedding model"):
        load_knowledge_base(other)


def test_chunking_change_triggers_full_rebuild(kb, corpus, embedder):
    changed = corpus.with_overrides(chunk_size=900)
    report = sync_knowledge_base(changed, embedder)
    assert report.changed and len(report.added) == 3
    assert load_knowledge_base(changed).manifest.chunking["chunk_size"] == 900


def test_corrupt_index_detected(kb, corpus):
    (corpus.index_dir / "versions" / kb.version / "chunks.json").write_text("[]")
    with pytest.raises(IndexCorruptError, match="inconsistent"):
        load_knowledge_base(corpus)


def test_truncated_index_file_detected(kb, corpus):
    (corpus.index_dir / "versions" / kb.version / "index.faiss").write_bytes(b"\x00" * 10)
    with pytest.raises(IndexCorruptError):
        load_knowledge_base(corpus)


def test_failed_write_keeps_previous_index(kb, corpus, embedder, monkeypatch):
    write_sds(corpus, "benzene_sds.pdf", "Benzene", "71-43-2", "-11 C")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(indexing.json, "dumps", boom)
    with pytest.raises(OSError):
        sync_knowledge_base(corpus, embedder)
    monkeypatch.undo()
    assert read_current_version(corpus.index_dir) == kb.version
    assert len(load_knowledge_base(corpus).manifest.documents) == 3
    assert not list((corpus.index_dir / "versions").glob(".tmp-*"))


def test_old_versions_pruned(kb, corpus, embedder):
    for flash in ("1 C", "2 C", "3 C"):
        write_sds(corpus, "ethanol_sds.pdf", "Ethanol", "64-17-5", flash)
        sync_knowledge_base(corpus, embedder)
    versions = [p for p in (corpus.index_dir / "versions").iterdir() if p.is_dir()]
    assert len(versions) == indexing.KEEP_VERSIONS


def test_write_lock_blocks_concurrent_writers(settings):
    entered = threading.Event()
    release = threading.Event()

    def holder():
        with index_write_lock(settings.index_dir):
            entered.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    entered.wait(5)
    try:
        with pytest.raises(IndexBusyError), index_write_lock(settings.index_dir):
            pass
    finally:
        release.set()
        t.join()
    with index_write_lock(settings.index_dir):  # free again
        pass


def test_lock_file_from_other_process_blocks(settings):
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / ".write.lock").write_text("12345 0")
    with pytest.raises(IndexBusyError), index_write_lock(settings.index_dir, timeout_s=0.3):
        pass


def test_diff_sources():
    from rag.chemicals import make_identity
    from rag.indexing import DocumentRecord, Manifest

    def rec(src, sha):
        return DocumentRecord(src, sha, 1, 1, make_identity("X"), 1, [])

    m = Manifest(2, "p", "t", {}, {}, [rec("a.pdf", "1"), rec("b.pdf", "2"), rec("c.pdf", "3")], 3)
    current = {"a.pdf": ("1", 1), "b.pdf": ("22", 1), "c2.pdf": ("3", 1), "d.pdf": ("4", 1)}
    diff = diff_sources(m, current)
    assert diff.changed == ["b.pdf"] and diff.added == ["d.pdf"]
    assert diff.renamed == [("c.pdf", "c2.pdf")] and diff.removed == []
    assert not diff.is_clean and "1 new" in diff.describe()


def test_status_reports_stale_documents(kb, corpus):
    assert assess_status(corpus, kb, None).state is KBState.READY
    write_sds(corpus, "uploads/benzene.pdf", "Benzene", "71-43-2", "-11 C")
    status = assess_status(corpus, kb, None)
    assert status.state is KBState.STALE and status.usable and "1 new" in status.message
    assert assess_status(corpus, None, IndexIncompatibleError("x")).state is KBState.INCOMPATIBLE
