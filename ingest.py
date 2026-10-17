"""
ingest.py  —  Build or update the knowledge-base index from SDS PDFs.

Usage:
    python ingest.py                 # incremental sync (only new/changed PDFs are embedded)
    python ingest.py --rebuild       # rebuild everything from scratch
    python ingest.py --check         # report index status without changing anything

Options:
    --data-dir PATH   source PDF directory   (default: DOCS_DIR or ./data/)
    --out-dir  PATH   index directory        (default: INDEX_DIR or ./storage/index/)

The app and this script share the same processing code (rag/), so documents
are parsed, chunked and embedded identically however they are added.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import configure_logging, load_settings
from rag.errors import ConfigError, IndexBusyError, KnowledgeBaseError
from rag.indexing import (
    KBState,
    assess_status,
    load_knowledge_base,
    sync_knowledge_base,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or update the SDS knowledge-base index.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing SDS PDF files")
    parser.add_argument("--out-dir", type=Path, help="Output directory for the index")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rebuild", action="store_true", help="Re-embed every document")
    mode.add_argument("--check", action="store_true", help="Only report status; exit 1 if not ready")
    return parser.parse_args()


_last_stage: list[str] = []


def _progress(stage: str, done: int, total: int) -> None:
    labels = {"scan": "Scanning", "parse": "Parsing", "embed": "Embedding", "write": "Writing"}
    if _last_stage and _last_stage[-1] != stage:
        print()
    _last_stage[:] = [stage]
    print(f"\r  {labels.get(stage, stage):<10} {done}/{total}", end="", flush=True)


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings()
        overrides: dict[str, object] = {}
        if args.data_dir:
            overrides["docs_dir"] = args.data_dir.resolve()
        if args.out_dir:
            overrides["index_dir"] = args.out_dir.resolve()
        if overrides:
            settings = settings.with_overrides(**overrides)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    print("=" * 60)
    print("  Chemical Safety RAG Assistant: knowledge-base ingestion")
    print("=" * 60)
    print(f"  Documents : {settings.docs_dir}")
    print(f"  Index     : {settings.index_dir}")
    print(f"  Embedding : {settings.embedding_model}")
    print(f"  Chunking  : {settings.chunk_size} chars, {settings.chunk_overlap} overlap\n")

    if not settings.docs_dir.exists():
        print(f"Documents directory not found: {settings.docs_dir}", file=sys.stderr)
        return 1

    if args.check:
        kb, err = None, None
        try:
            kb = load_knowledge_base(settings)
        except KnowledgeBaseError as exc:
            err = exc
        status = assess_status(settings, kb, err)
        print(f"Status: {status.state.value}")
        if status.message:
            print(f"  {status.message}")
        for issue in status.issues:
            print(f"  [{issue.level.value}] {issue.message}")
        if kb:
            for d in kb.manifest.documents:
                print(f"  - {d.source}: {d.chemical.name} (CAS {d.chemical.cas or 'n/a'}), "
                      f"{d.pages} pages, {d.chunk_count} chunks")
        return 0 if status.state is KBState.READY else 1

    from rag.embeddings import Embedder

    print("Loading embedding model…")
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    try:
        report = sync_knowledge_base(settings, embedder, full_rebuild=args.rebuild,
                                     progress=_progress, lock_timeout_s=10)
    except IndexBusyError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print("\n")
    if not report.changed:
        print(f"Index is up to date ({report.summary()}).")
    else:
        print(f"Index version {report.version} written in {report.duration_s:.1f}s: {report.summary()}")
    for src, msg in report.failed:
        print(f"  ! {src}: {msg}")
    for src, original in report.duplicates:
        print(f"  = {src}: same content as {original}; not indexed twice")
    print("\nRun `streamlit run app.py` to start the assistant.")
    return 0 if not report.failed else 3


if __name__ == "__main__":
    sys.exit(main())
