from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings
from rag.indexing import KnowledgeBase, load_knowledge_base, sync_knowledge_base
from tests.helpers import FakeEmbedder, make_pdf, sds_pages


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir()
    s = Settings(
        docs_dir=docs,
        index_dir=tmp_path / "index",
        embedding_model="test/fake-embedder",
        chunk_size=600,
        chunk_overlap=80,
        top_k=4,
        candidate_k=20,
    )
    s.validate()
    return s


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def write_sds(settings: Settings, filename: str, name: str, cas: str, flash: str, **kw) -> Path:
    path = settings.docs_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_pdf(sds_pages(name, cas, flash, **kw), title=f"Safety Data Sheet: {name}"))
    return path


@pytest.fixture
def corpus(settings: Settings) -> Settings:
    """Three synthetic SDSs with deliberately similar wording."""
    write_sds(settings, "acetone_sds.pdf", "Acetone", "67-64-1", "-17 C",
              synonym="2-Propanone", extra={14: ["UN number: UN 1090"]})
    write_sds(settings, "ethanol_sds.pdf", "Ethanol", "64-17-5", "12 C",
              synonym="Ethyl alcohol", extra={14: ["UN number: UN 1170"]})
    write_sds(settings, "methanol_sds.pdf", "Methanol", "67-56-1", "9.7 C",
              synonym="Methyl alcohol", extra={14: ["UN number: UN 1230"], 7: ["No data available"]})
    return settings


@pytest.fixture
def kb(corpus: Settings, embedder: FakeEmbedder) -> KnowledgeBase:
    sync_knowledge_base(corpus, embedder)
    return load_knowledge_base(corpus)
