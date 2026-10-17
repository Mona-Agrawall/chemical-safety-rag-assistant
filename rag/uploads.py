"""Validate and stage user uploads into the documents directory, with rollback."""

from __future__ import annotations

import contextlib
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from rag.config import Settings
from rag.errors import DocumentError
from rag.indexing import Manifest
from rag.loader import parse_pdf_bytes, sha256_file


class UploadStatus(str, Enum):
    ADDED = "added"
    REPLACED = "replaced"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


@dataclass
class UploadOutcome:
    filename: str
    status: UploadStatus
    message: str
    saved_path: Path | None = None
    backup_path: Path | None = None


@dataclass
class StagedUploads:
    outcomes: list[UploadOutcome] = field(default_factory=list)

    @property
    def saved(self) -> list[UploadOutcome]:
        return [o for o in self.outcomes if o.saved_path is not None]

    def commit(self, outcomes: list[UploadOutcome] | None = None) -> None:
        """Indexing succeeded: discard backups of replaced files."""
        for o in self.saved if outcomes is None else outcomes:
            if o.backup_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    o.backup_path.unlink()
                o.backup_path = None

    def rollback(self, outcomes: list[UploadOutcome] | None = None) -> None:
        """Indexing failed: remove new files and restore replaced ones."""
        for o in self.saved if outcomes is None else outcomes:
            assert o.saved_path is not None
            if o.backup_path is not None and o.backup_path.exists():
                os.replace(o.backup_path, o.saved_path)
            else:
                with contextlib.suppress(FileNotFoundError):
                    o.saved_path.unlink()

    def settle(self, docs_dir: Path, failed: dict[str, str]) -> tuple[list[UploadOutcome], list[UploadOutcome]]:
        """After a sync: keep uploads that were indexed, roll back those that failed.

        ``failed`` maps source paths (relative to ``docs_dir``) to error messages.
        Only the failed files are rolled back; rolling back everything would delete
        files the new index already contains. Returns ``(kept, rolled_back)``.
        """
        kept: list[UploadOutcome] = []
        rolled_back: list[UploadOutcome] = []
        for o in self.saved:
            assert o.saved_path is not None
            src = o.saved_path.relative_to(docs_dir).as_posix()
            if src in failed:
                o.message = failed[src]
                rolled_back.append(o)
            else:
                kept.append(o)
        self.commit(kept)
        self.rollback(rolled_back)
        return kept, rolled_back


def sanitize_filename(name: str) -> str:
    """Safe, portable file name ending in .pdf (no paths, no hidden files)."""
    base = Path(name.replace("\\", "/")).name
    stem = base[:-4] if base.lower().endswith(".pdf") else base
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:100] or "document"
    return f"{stem}.pdf"


def stage_uploads(settings: Settings, files: list[tuple[str, bytes]], manifest: Manifest | None) -> StagedUploads:
    """Validate each upload and write accepted ones into ``uploads_dir``.

    Writes are atomic (temp file + rename). A file with the same name as an
    earlier upload replaces it (the old copy is kept as a hidden backup until
    :meth:`StagedUploads.commit`). Content already in the knowledge base is
    reported as a duplicate and not written again.
    """
    staged = StagedUploads()
    known = {d.sha256: d.source for d in (manifest.documents if manifest else [])}
    max_bytes = settings.max_upload_mb * 1024 * 1024
    upload_dir = settings.uploads_dir
    seen: set[str] = set()

    for original_name, data in files:
        safe = sanitize_filename(original_name)
        try:
            doc = parse_pdf_bytes(
                data, f"{settings.uploads_subdir}/{safe}", max_bytes=max_bytes, max_pages=settings.max_pdf_pages
            )
        except DocumentError as exc:
            staged.outcomes.append(UploadOutcome(original_name, UploadStatus.REJECTED, str(exc)))
            continue

        sha = doc.sha256
        if sha in known:
            staged.outcomes.append(UploadOutcome(
                original_name, UploadStatus.DUPLICATE, f"Already in the knowledge base as {known[sha]}."
            ))
            continue
        if sha in seen:
            staged.outcomes.append(UploadOutcome(
                original_name, UploadStatus.DUPLICATE, "Same content as another file in this upload."
            ))
            continue
        seen.add(sha)

        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / safe
        backup = None
        status = UploadStatus.ADDED
        if target.exists():
            if sha256_file(target) == sha:
                staged.outcomes.append(UploadOutcome(
                    original_name, UploadStatus.DUPLICATE, f"{safe} is already saved (not yet indexed)."
                ))
                seen.discard(sha)
                continue
            backup = upload_dir / f".backup-{int(time.time() * 1000)}-{safe}"
            os.replace(target, backup)
            status = UploadStatus.REPLACED
        tmp = upload_dir / f".{safe}.part"
        tmp.write_bytes(data)
        os.replace(tmp, target)
        msg = "Replaced the previous version." if status is UploadStatus.REPLACED else "Saved."
        staged.outcomes.append(UploadOutcome(original_name, status, msg, target, backup))
    return staged
