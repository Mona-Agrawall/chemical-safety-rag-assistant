"""Live end-to-end test against a real Ollama server (skipped when unavailable).

Run explicitly with:  pytest -m live
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import BASE_DIR, load_settings
from rag.health import check_ollama

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_pipeline(tmp_path_factory):
    settings = load_settings()
    health = check_ollama(settings.ollama_base_url, settings.ollama_model, settings.health_timeout_s)
    if not health.ok:
        pytest.skip(f"Ollama not available: {health.message}")
    from rag.embeddings import Embedder
    from rag.indexing import load_knowledge_base, sync_knowledge_base
    from rag.pipeline import RagPipeline, make_llm

    s = settings.with_overrides(docs_dir=BASE_DIR / "data",
                                index_dir=Path(tmp_path_factory.mktemp("live")) / "index")
    embedder = Embedder(s.embedding_model, s.embedding_device)
    sync_knowledge_base(s, embedder)
    return RagPipeline(s, load_knowledge_base(s), embedder, make_llm(s))


def test_live_answer_is_correct_and_cited(live_pipeline):
    ans = live_pipeline.ask("What is the flash point of acetone?")
    assert "-17" in ans.text.replace("−", "-")
    assert ans.cited_sources, ans.text
    assert all(s.filename == "acetone_sds.pdf" for s in ans.cited_sources)
    assert any(s.page_start <= 8 <= s.page_end for s in ans.cited_sources)
    assert not ans.invalid_labels


def test_live_unindexed_chemical(live_pipeline):
    ans = live_pipeline.ask("What is the flash point of toluene?")
    assert ans.refused and not ans.cited_sources
