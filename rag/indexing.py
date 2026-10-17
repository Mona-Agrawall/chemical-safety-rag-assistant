"""Knowledge-base persistence: build, incremental sync, manifest, atomic swap.

On-disk layout (``INDEX_DIR``)::

    CURRENT                     # name of the active version directory
    versions/<version>/
        manifest.json           # provenance + configuration of this build
        chunks.json             # chunk text and metadata, row-aligned with the index
        index.faiss             # FAISS IndexFlatIP over normalised embeddings

A sync writes a complete new version directory and then atomically replaces
``CURRENT``. Readers therefore always see either the old or the new index,
never a half-written one, and a failed build leaves the last good index intact.
No pickle is used anywhere, so loading an index cannot execute code.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rag.chemicals import ChemicalIdentity, extract_identity
from rag.chunking import Chunk, chunk_document
from rag.config import INDEX_FORMAT_VERSION, PIPELINE_VERSION, Settings
from rag.errors import (
    DocumentError,
    IndexBusyError,
    IndexCorruptError,
    IndexIncompatibleError,
    IndexMissingError,
)
from rag.loader import PdfDocument, discover_pdfs, load_pdf, relative_source, sha256_file

if TYPE_CHECKING:
    import faiss

    from rag.embeddings import Embedder

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]
KEEP_VERSIONS = 2
LOCK_STALE_S = 30 * 60


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DocumentRecord:
    source: str
    sha256: str
    size_bytes: int
    pages: int
    chemical: ChemicalIdentity
    chunk_count: int
    sections: list[int]
    title: str = ""

    @property
    def filename(self) -> str:
        return Path(self.source).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "pages": self.pages,
            "chemical": self.chemical.to_dict(),
            "chunk_count": self.chunk_count,
            "sections": self.sections,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DocumentRecord:
        return cls(
            source=d["source"],
            sha256=d["sha256"],
            size_bytes=int(d["size_bytes"]),
            pages=int(d["pages"]),
            chemical=ChemicalIdentity.from_dict(d["chemical"]),
            chunk_count=int(d["chunk_count"]),
            sections=list(d.get("sections", [])),
            title=d.get("title", ""),
        )


@dataclass
class Manifest:
    format_version: int
    pipeline_version: str
    created_at: str
    embedding: dict[str, Any]
    chunking: dict[str, Any]
    documents: list[DocumentRecord]
    chunk_count: int
    libraries: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at,
            "embedding": self.embedding,
            "chunking": self.chunking,
            "chunk_count": self.chunk_count,
            "libraries": self.libraries,
            "documents": [d.to_dict() for d in self.documents],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        return cls(
            format_version=int(d["format_version"]),
            pipeline_version=d["pipeline_version"],
            created_at=d["created_at"],
            embedding=d["embedding"],
            chunking=d["chunking"],
            documents=[DocumentRecord.from_dict(x) for x in d["documents"]],
            chunk_count=int(d["chunk_count"]),
            libraries=d.get("libraries", {}),
        )


class IssueLevel(str, Enum):
    ERROR = "error"  # index must not be used
    WARNING = "warning"  # usable, but a rebuild/sync is recommended


@dataclass(frozen=True)
class Issue:
    level: IssueLevel
    message: str


def check_compatibility(manifest: Manifest, settings: Settings) -> list[Issue]:
    issues: list[Issue] = []
    if manifest.format_version != INDEX_FORMAT_VERSION:
        issues.append(Issue(IssueLevel.ERROR, (
            f"Index format v{manifest.format_version} is not supported "
            f"(expected v{INDEX_FORMAT_VERSION}). Rebuild the index."
        )))
    if manifest.embedding.get("model") != settings.embedding_model:
        issues.append(Issue(IssueLevel.ERROR, (
            f"Index was built with embedding model '{manifest.embedding.get('model')}' but "
            f"EMBEDDING_MODEL is '{settings.embedding_model}'. Query vectors would not match; "
            "rebuild the index."
        )))
    if manifest.pipeline_version != PIPELINE_VERSION:
        issues.append(Issue(IssueLevel.WARNING, (
            f"Index was built by pipeline '{manifest.pipeline_version}', current is "
            f"'{PIPELINE_VERSION}'. Rebuild to pick up chunking improvements."
        )))
    ch = manifest.chunking
    if ch.get("chunk_size") != settings.chunk_size or ch.get("chunk_overlap") != settings.chunk_overlap:
        issues.append(Issue(IssueLevel.WARNING, (
            f"Index chunking ({ch.get('chunk_size')}/{ch.get('chunk_overlap')}) differs from "
            f"configuration ({settings.chunk_size}/{settings.chunk_overlap}). Rebuild to apply."
        )))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Loaded knowledge base
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class KnowledgeBase:
    version: str
    manifest: Manifest
    chunks: list[Chunk]
    index: faiss.Index

    @property
    def identities(self) -> list[ChemicalIdentity]:
        seen: dict[str, ChemicalIdentity] = {}
        for d in self.manifest.documents:
            seen.setdefault(d.chemical.key, d.chemical)
        return sorted(seen.values(), key=lambda i: i.name.lower())

    def documents_for(self, chemical_key: str) -> list[DocumentRecord]:
        return [d for d in self.manifest.documents if d.chemical.key == chemical_key]


def _versions_dir(index_dir: Path) -> Path:
    return index_dir / "versions"


def read_current_version(index_dir: Path) -> str | None:
    try:
        name = (index_dir / "CURRENT").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise IndexCorruptError(f"Cannot read {index_dir / 'CURRENT'}: {exc}") from exc
    return name or None


def load_knowledge_base(settings: Settings, version: str | None = None) -> KnowledgeBase:
    """Load the active (or a specific) index version and validate it."""
    import faiss

    index_dir = settings.index_dir
    version = version or read_current_version(index_dir)
    if not version:
        raise IndexMissingError(f"No index found in {index_dir}. Run `python ingest.py` to build it.")
    vdir = _versions_dir(index_dir) / version
    try:
        manifest = Manifest.from_dict(json.loads((vdir / "manifest.json").read_text(encoding="utf-8")))
        raw_chunks = json.loads((vdir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(c) for c in raw_chunks]
        data = np.frombuffer((vdir / "index.faiss").read_bytes(), dtype="uint8")
        index = faiss.deserialize_index(data)
    except FileNotFoundError as exc:
        raise IndexCorruptError(f"Index version {version} is incomplete: {exc.filename} missing.") from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise IndexCorruptError(f"Index version {version} is unreadable: {exc}") from exc

    if index.ntotal != len(chunks) or manifest.chunk_count != len(chunks):
        raise IndexCorruptError(
            f"Index version {version} is inconsistent: {index.ntotal} vectors, "
            f"{len(chunks)} chunks, manifest says {manifest.chunk_count}."
        )
    if index.d != int(manifest.embedding.get("dimension", index.d)):
        raise IndexCorruptError("Index vector dimension does not match the manifest.")
    errors = [i.message for i in check_compatibility(manifest, settings) if i.level is IssueLevel.ERROR]
    if errors:
        raise IndexIncompatibleError(" ".join(errors))
    return KnowledgeBase(version=version, manifest=manifest, chunks=chunks, index=index)


# ─────────────────────────────────────────────────────────────────────────────
# Source scanning and diffing
# ─────────────────────────────────────────────────────────────────────────────
_hash_cache: dict[tuple[str, int, int], str] = {}
_hash_lock = threading.Lock()


def scan_sources(docs_dir: Path) -> dict[str, tuple[str, int]]:
    """Map relative source path -> (sha256, size) for every PDF under docs_dir."""
    out: dict[str, tuple[str, int]] = {}
    for p in discover_pdfs(docs_dir):
        st = p.stat()
        key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
        with _hash_lock:
            sha = _hash_cache.get(key)
        if sha is None:
            sha = sha256_file(p)
            with _hash_lock:
                _hash_cache[key] = sha
        out[relative_source(p, docs_dir)] = (sha, st.st_size)
    return out


@dataclass
class SourceDiff:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)  # (old, new)
    # (copy, indexed original): same content as an indexed file, deliberately not indexed again.
    duplicates: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.added or self.changed or self.removed or self.renamed)

    def describe(self) -> str:
        parts = []
        if self.added:
            parts.append(f"{len(self.added)} new")
        if self.changed:
            parts.append(f"{len(self.changed)} modified")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.renamed:
            parts.append(f"{len(self.renamed)} renamed")
        return ", ".join(parts) or "no changes"


def diff_sources(manifest: Manifest | None, current: dict[str, tuple[str, int]]) -> SourceDiff:
    indexed = {d.source: d.sha256 for d in (manifest.documents if manifest else [])}
    diff = SourceDiff()
    by_sha_old = {sha: src for src, sha in indexed.items()}
    renamed_old: set[str] = set()
    for src, (sha, _) in sorted(current.items()):
        if src in indexed:
            if indexed[src] != sha:
                diff.changed.append(src)
        elif sha in by_sha_old and by_sha_old[sha] not in current and by_sha_old[sha] not in renamed_old:
            renamed_old.add(by_sha_old[sha])
            diff.renamed.append((by_sha_old[sha], src))
        elif sha in by_sha_old:
            # Regression: a byte-identical copy used to be reported as "new" forever.
            diff.duplicates.append((src, by_sha_old[sha]))
        else:
            diff.added.append(src)
    diff.removed = [src for src in indexed if src not in current and src not in renamed_old]
    return diff


# ─────────────────────────────────────────────────────────────────────────────
# Locking
# ─────────────────────────────────────────────────────────────────────────────
_process_lock = threading.Lock()


@contextlib.contextmanager
def index_write_lock(index_dir: Path, timeout_s: float = 0.0) -> Iterator[None]:
    """Exclusive writer lock across threads (Streamlit sessions) and processes."""
    acquired = (
        _process_lock.acquire(timeout=timeout_s)
        if timeout_s > 0
        else _process_lock.acquire(blocking=False)
    )
    if not acquired:
        raise IndexBusyError("The knowledge base is already being updated. Try again shortly.")
    lock_path = index_dir / ".write.lock"
    fd = None
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time():.0f}".encode())
                break
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > LOCK_STALE_S:
                    log.warning("Removing stale index lock (%.0f s old)", age)
                    with contextlib.suppress(FileNotFoundError):
                        lock_path.unlink()
                    continue
                if time.monotonic() >= deadline:
                    raise IndexBusyError(
                        "Another process is updating the knowledge base. Try again shortly."
                    ) from None
                time.sleep(0.2)
        yield
    finally:
        if fd is not None:
            os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                lock_path.unlink()
        _process_lock.release()


# ─────────────────────────────────────────────────────────────────────────────
# Building / syncing
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SyncReport:
    version: str | None
    changed: bool
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # (copy, indexed original)
    chunk_count: int = 0
    document_count: int = 0
    duration_s: float = 0.0

    def summary(self) -> str:
        bits = [f"{self.document_count} documents, {self.chunk_count} chunks"]
        for label, items in (("added", self.added), ("updated", self.updated),
                             ("removed", self.removed), ("renamed", self.renamed)):
            if items:
                bits.append(f"{len(items)} {label}")
        if self.failed:
            bits.append(f"{len(self.failed)} failed")
        if self.duplicates:
            bits.append(f"{len(self.duplicates)} duplicate(s) skipped")
        return ", ".join(bits)


def process_document(doc: PdfDocument, settings: Settings) -> tuple[DocumentRecord, list[Chunk]]:
    identity = extract_identity(doc.pages, doc.title, doc.filename)
    chunks = chunk_document(
        doc.pages, identity, doc.source, doc.sha256, settings.chunk_size, settings.chunk_overlap
    )
    if not chunks:
        raise DocumentError(f"{doc.filename}: no usable text after processing.")
    record = DocumentRecord(
        source=doc.source,
        sha256=doc.sha256,
        size_bytes=doc.size_bytes,
        pages=doc.page_count,
        chemical=identity,
        chunk_count=len(chunks),
        sections=sorted({c.section for c in chunks if c.section > 0}),
        title=doc.title,
    )
    return record, chunks


def _library_versions() -> dict[str, str]:
    """Installed versions of the libraries that shape the index (no imports needed)."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for dist in ("faiss-cpu", "sentence-transformers", "transformers", "torch", "pypdf", "numpy"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = "not installed"
    return out


def _write_version(index_dir: Path, manifest: Manifest, chunks: list[Chunk], vectors: np.ndarray) -> str:
    import faiss

    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    vroot = _versions_dir(index_dir)
    vroot.mkdir(parents=True, exist_ok=True)
    tmp = vroot / f".tmp-{version}"
    tmp.mkdir()
    try:
        index = faiss.IndexFlatIP(int(manifest.embedding["dimension"]))
        if len(vectors):
            index.add(np.ascontiguousarray(vectors, dtype="float32"))
        (tmp / "index.faiss").write_bytes(faiss.serialize_index(index).tobytes())
        (tmp / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False), encoding="utf-8"
        )
        (tmp / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.rename(vroot / version)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    # Atomically point CURRENT at the new version.
    pointer_tmp = index_dir / f".CURRENT.{version}.tmp"
    pointer_tmp.write_text(version, encoding="utf-8")
    os.replace(pointer_tmp, index_dir / "CURRENT")
    return version


def _prune_versions(index_dir: Path, keep: str) -> None:
    vroot = _versions_dir(index_dir)
    if not vroot.exists():
        return
    versions = sorted((p for p in vroot.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=lambda p: p.name)
    old = [p for p in versions if p.name != keep][: max(0, len(versions) - KEEP_VERSIONS)]
    for p in old:
        shutil.rmtree(p, ignore_errors=True)
    for p in vroot.glob(".tmp-*"):
        # Leftovers of crashed builds older than an hour.
        if time.time() - p.stat().st_mtime > 3600:
            shutil.rmtree(p, ignore_errors=True)


def sync_knowledge_base(
    settings: Settings,
    embedder: Embedder,
    *,
    full_rebuild: bool = False,
    progress: ProgressFn | None = None,
    lock_timeout_s: float = 0.0,
) -> SyncReport:
    """Bring the index in line with the documents directory.

    Unchanged documents (same SHA-256) reuse their stored chunks and vectors;
    new or modified documents are re-parsed and embedded; deleted documents
    are dropped; renamed documents keep their vectors under the new name.
    """
    t0 = time.perf_counter()
    notify = progress or (lambda stage, done, total: None)
    with index_write_lock(settings.index_dir, lock_timeout_s):
        previous: KnowledgeBase | None = None
        if not full_rebuild:
            try:
                previous = load_knowledge_base(settings)
                if any(i.level is IssueLevel.WARNING for i in check_compatibility(previous.manifest, settings)):
                    log.info("Index settings changed; performing a full rebuild.")
                    previous = None
            except IndexMissingError:
                previous = None
            except (IndexIncompatibleError, IndexCorruptError) as exc:
                log.warning("Existing index unusable (%s); rebuilding from scratch.", exc)
                previous = None

        notify("scan", 0, 1)
        current = scan_sources(settings.docs_dir)
        diff = diff_sources(previous.manifest if previous else None, current)
        report = SyncReport(version=previous.version if previous else None, changed=False)

        if previous is not None and diff.is_clean:
            report.duplicates = list(diff.duplicates)
            report.unchanged = [d.source for d in previous.manifest.documents]
            report.chunk_count = len(previous.chunks)
            report.document_count = len(previous.manifest.documents)
            report.duration_s = time.perf_counter() - t0
            return report

        # Reuse stored vectors for documents whose content is unchanged.
        reuse: dict[str, tuple[DocumentRecord, list[Chunk], np.ndarray]] = {}
        if previous is not None:
            all_vecs = previous.index.reconstruct_n(0, previous.index.ntotal)
            by_sha: dict[str, list[int]] = {}
            for i, c in enumerate(previous.chunks):
                by_sha.setdefault(c.doc_sha, []).append(i)
            for rec in previous.manifest.documents:
                rows = by_sha.get(rec.sha256, [])
                reuse[rec.sha256] = (rec, [previous.chunks[i] for i in rows], all_vecs[rows])

        records: list[DocumentRecord] = []
        chunks: list[Chunk] = []
        vectors: list[np.ndarray] = []
        to_embed: list[tuple[DocumentRecord, list[Chunk]]] = []
        renamed_new = {new: old for old, new in diff.renamed}
        seen_shas: dict[str, str] = {}  # sha -> source it is indexed under

        # Indexed originals first, so a new copy never displaces the file already indexed.
        prior = {d.source for d in previous.manifest.documents} if previous else set()
        sources = sorted(current, key=lambda s: (s not in prior, s))
        for n, src in enumerate(sources, start=1):
            sha, _ = current[src]
            notify("parse", n, len(sources))
            if sha in seen_shas:
                report.duplicates.append((src, seen_shas[sha]))
                continue
            if sha in reuse:
                rec, rchunks, rvecs = reuse[sha]
                if rec.source != src:
                    rec = replace(rec, source=src)
                    rchunks = [replace(c, source=src) for c in rchunks]
                records.append(rec)
                chunks.extend(rchunks)
                vectors.append(rvecs)
                seen_shas[sha] = src
                if src in renamed_new:
                    report.renamed.append((renamed_new[src], src))
                else:
                    report.unchanged.append(src)
                continue
            try:
                doc = load_pdf(settings.docs_dir / src, settings.docs_dir, max_pages=settings.max_pdf_pages)
                rec, dchunks = process_document(doc, settings)
            except DocumentError as exc:
                log.warning("Skipping %s: %s", src, exc)
                report.failed.append((src, str(exc)))
                continue
            seen_shas[sha] = src
            to_embed.append((rec, dchunks))
            (report.updated if src in diff.changed else report.added).append(src)

        total_new = sum(len(c) for _, c in to_embed)
        done = 0
        for rec, dchunks in to_embed:
            def _p(d: int, t: int, base: int = done) -> None:
                notify("embed", base + d, total_new)
            vecs = embedder.embed_documents([c.search_text for c in dchunks], progress=_p)
            done += len(dchunks)
            records.append(rec)
            chunks.extend(dchunks)
            vectors.append(vecs)

        report.removed = list(diff.removed)
        if previous is not None and not (report.added or report.updated or report.removed or report.renamed):
            # Only failures (e.g. an unreadable new file): keep the current index.
            report.unchanged = [d.source for d in previous.manifest.documents]
            report.chunk_count = len(previous.chunks)
            report.document_count = len(previous.manifest.documents)
            report.duration_s = time.perf_counter() - t0
            return report

        # Stable ordering by source path; `chunks` and `vectors` are row-aligned.
        records.sort(key=lambda r: r.source.lower())
        flat_vecs = np.vstack(vectors) if vectors else np.zeros((0, embedder.dimension), dtype="float32")
        rows_by_sha = _rows_by_sha(chunks)
        order = [i for rec in records for i in rows_by_sha[rec.sha256]]
        final_chunks = [chunks[i] for i in order]
        stacked = flat_vecs[order] if order else flat_vecs

        notify("write", 0, 1)
        manifest = Manifest(
            format_version=INDEX_FORMAT_VERSION,
            pipeline_version=PIPELINE_VERSION,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            embedding={
                "model": settings.embedding_model,
                "revision": embedder.revision,
                "dimension": embedder.dimension,
                "query_prefix": embedder.query_prefix,
                "passage_prefix": embedder.passage_prefix,
                "normalized": True,
                "metric": "inner_product",
            },
            chunking={
                "strategy": PIPELINE_VERSION,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            documents=records,
            chunk_count=len(final_chunks),
            libraries=_library_versions(),
        )
        version = _write_version(settings.index_dir, manifest, final_chunks, stacked)
        _prune_versions(settings.index_dir, keep=version)
        notify("write", 1, 1)

        report.version = version
        report.changed = True
        report.chunk_count = len(final_chunks)
        report.document_count = len(records)
        report.duration_s = time.perf_counter() - t0
        log.info("Index version %s written: %s (%.1fs)", version, report.summary(), report.duration_s)
        return report


def _rows_by_sha(chunks: list[Chunk]) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        rows.setdefault(c.doc_sha, []).append(i)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Status for the UI / CLI
# ─────────────────────────────────────────────────────────────────────────────
class KBState(str, Enum):
    READY = "ready"
    STALE = "stale"  # usable; documents directory differs from the index
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    CORRUPT = "corrupt"


@dataclass
class KBStatus:
    state: KBState
    message: str = ""
    issues: list[Issue] = field(default_factory=list)
    diff: SourceDiff | None = None

    @property
    def usable(self) -> bool:
        return self.state in (KBState.READY, KBState.STALE)


def assess_status(settings: Settings, kb: KnowledgeBase | None, load_error: Exception | None) -> KBStatus:
    if kb is None:
        if isinstance(load_error, IndexIncompatibleError):
            return KBStatus(KBState.INCOMPATIBLE, str(load_error))
        if isinstance(load_error, IndexCorruptError):
            return KBStatus(KBState.CORRUPT, str(load_error))
        return KBStatus(KBState.MISSING, str(load_error) if load_error else "No index has been built yet.")
    issues = check_compatibility(kb.manifest, settings)
    diff = diff_sources(kb.manifest, scan_sources(settings.docs_dir))
    if not diff.is_clean:
        return KBStatus(KBState.STALE, f"Documents folder changed since indexing: {diff.describe()}.",
                        issues, diff)
    if issues:
        return KBStatus(KBState.STALE, issues[0].message, issues, diff)
    return KBStatus(KBState.READY, "", issues, diff)
