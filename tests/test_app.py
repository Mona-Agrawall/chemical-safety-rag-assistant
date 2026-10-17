"""UI tests: run the real app.py with Streamlit's AppTest, fakes for models."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from rag.errors import LLMUnavailableError
from rag.health import HealthState, HealthStatus
from tests.helpers import FakeEmbedder, FakeLLM

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class Env:
    def __init__(self) -> None:
        self.health = HealthStatus(HealthState.OK, "mistral ready", ["mistral:latest"])
        self.llm = FakeLLM("The flash point is 9.7 C [S1].")


@pytest.fixture
def env(monkeypatch, corpus) -> Env:
    e = Env()
    monkeypatch.setenv("DOCS_DIR", str(corpus.docs_dir))
    monkeypatch.setenv("INDEX_DIR", str(corpus.index_dir))
    monkeypatch.setenv("EMBEDDING_MODEL", "test/fake-embedder")
    monkeypatch.setenv("CHUNK_SIZE", "600")
    monkeypatch.setenv("CHUNK_OVERLAP", "80")
    monkeypatch.setattr("rag.embeddings.Embedder", lambda model, device="cpu": FakeEmbedder(model))
    monkeypatch.setattr("rag.health.check_ollama", lambda *a, **k: e.health)
    monkeypatch.setattr("rag.pipeline.make_llm", lambda s: e.llm)
    st.cache_resource.clear()
    yield e
    st.cache_resource.clear()


def run(at: AppTest | None = None) -> AppTest:
    at = at or AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


def test_first_run_builds_index_and_lists_documents(env, corpus):
    at = run()
    at = run(at)  # after the automatic first build the app reruns
    text = all_text(at)
    assert "Ask your Safety Data Sheets" in text
    assert "3 documents" in text
    assert (corpus.index_dir / "CURRENT").exists()
    labels = [b.label for b in at.button]
    assert any("flash point of Acetone" in label for label in labels), labels
    assert not at.chat_input[0].disabled


def test_chat_answer_with_rendered_citation_and_sources(env):
    at = run(run())
    at.chat_input[0].set_value("What is the flash point of methanol?").run()
    assert not at.exception
    answers = [m for m in at.chat_message if m.name == "assistant"]
    assert answers
    md = "\n".join(str(x.value) for x in answers[-1].markdown)
    assert "9.7 C [1]" in md and "[S1]" not in md
    # Expanders with an icon are exposed as Status blocks by AppTest.
    assert any("Sources · 1 cited" in s.label for s in at.status)
    assert "Searching: Methanol" in all_text(at)
    # The prompt sent to the model only contained methanol excerpts.
    assert "Chemical: Methanol" in env.llm.calls[-1][-1]["content"]
    assert "Chemical: Acetone" not in env.llm.calls[-1][-1]["content"]


def test_example_question_button_asks_it(env):
    at = run(run())
    btn = next(b for b in at.button if "flash point of Acetone" in b.label)
    btn.click().run()
    assert not at.exception
    assert any(m.name == "user" for m in at.chat_message)


def test_model_offline_disables_chat_and_retry_recovers(env):
    env.health = HealthStatus(HealthState.SERVER_DOWN, "Cannot reach Ollama. Start it with `ollama serve`.")
    at = run(run())
    assert at.chat_input[0].disabled
    assert "ollama serve" in all_text(at)
    env.health = HealthStatus(HealthState.OK, "mistral ready", ["mistral:latest"])
    at.button(key="retry_health").click().run()
    at = run(at)
    assert not at.chat_input[0].disabled


def test_generation_failure_shows_error_without_crashing(env):
    env.llm = FakeLLM(error=LLMUnavailableError("Cannot connect to Ollama at http://localhost:11434."))
    at = run(run())
    at.chat_input[0].set_value("flash point of acetone?").run()
    assert not at.exception
    assert any("Could not generate an answer" in e.value for e in at.error)


def test_unindexed_chemical_gets_clear_answer(env):
    at = run(run())
    at.chat_input[0].set_value("What is the flash point of toluene?").run()
    assert "does not contain a Safety Data Sheet for Toluene" in all_text(at)
    assert env.llm.calls == []


def test_clear_chat_keeps_knowledge_base(env, corpus):
    at = run(run())
    at.chat_input[0].set_value("flash point of ethanol?").run()
    assert at.session_state["messages"]
    clear = next(b for b in at.button if b.label == "Clear conversation")
    clear.click().run()
    at = run(at)
    assert at.session_state["messages"] == []
    assert (corpus.index_dir / "CURRENT").exists()


def test_invalid_configuration_is_reported(env, monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    st.cache_resource.clear()
    at = run()
    assert any("Configuration error" in e.value for e in at.error)


def test_stale_index_offers_update(env, corpus):
    from tests.conftest import write_sds

    at = run(run())
    write_sds(corpus, "uploads/benzene.pdf", "Benzene", "71-43-2", "-11 C")
    at = run(at)
    assert "Documents folder changed since indexing" in all_text(at)
    at.button(key="sync_btn").click().run()
    at = run(at)
    assert "4 documents" in all_text(at)


def test_answer_interrupted_by_rerun_is_marked_not_silently_dropped(env):
    """Regression: a question whose streamed answer was cut off by a rerun was left without any reply."""
    at = run(run())
    at.session_state["messages"] = [{"role": "user", "content": "flash point of acetone?", "scope": ["acetone"]}]
    at = run(at)
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[-1]["error"]
    assert any("interrupted" in e.value for e in at.error)
    at.chat_input[0].set_value("flash point of ethanol?").run()
    assert not at.exception
    assert [m["role"] for m in at.session_state["messages"]] == ["user", "assistant", "user", "assistant"]


def test_export_and_clear_are_available_right_after_the_first_answer(env):
    """Regression: the sidebar was rendered before the answer was added, so after the first answer
    Export/Clear stayed disabled and every export lacked the most recent question and answer."""
    at = run(run())
    at.chat_input[0].set_value("What is the flash point of methanol?").run()
    assert not at.exception
    export = next(e for e in at.get("download_button") if e.proto.label == "Export as Markdown")
    assert not export.proto.disabled
    clear = next(b for b in at.button if b.label == "Clear conversation")
    assert not clear.disabled


def test_scope_filter_for_removed_chemical_does_not_crash(env):
    """Regression: a stored filter value that is no longer an option used to raise."""
    at = run(run())
    at.session_state["scope_filter"] = ["acetone", "no-longer-indexed"]
    at = run(at)
    assert at.session_state["scope_filter"] == ["acetone"]
