from pathlib import Path

import pytest

from rag.config import BASE_DIR, Settings, load_settings
from rag.errors import ConfigError

ENV_VARS = [
    "OLLAMA_BASE_URL", "OLLAMA_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "CANDIDATE_K",
    "HYBRID_SEARCH", "DOCS_DIR", "INDEX_DIR", "MAX_UPLOAD_MB", "LOG_LEVEL", "RERANKER_ENABLED",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    # Point at an empty .env so a developer's local file cannot influence tests.
    empty = tmp_path / "empty.env"
    empty.write_text("")
    return empty


def test_defaults_are_valid(clean_env):
    s = load_settings(clean_env)
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.docs_dir == BASE_DIR / "data"
    assert s.uploads_dir == BASE_DIR / "data" / "uploads"


def test_environment_overrides(monkeypatch, clean_env, tmp_path):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/")
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("HYBRID_SEARCH", "off")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("DOCS_DIR", "relative/docs")
    s = load_settings(clean_env)
    assert s.ollama_base_url == "http://ollama:11434"  # trailing slash stripped
    assert s.chunk_size == 1200
    assert s.hybrid_search is False
    assert s.index_dir == tmp_path / "idx"
    assert s.docs_dir == BASE_DIR / "relative" / "docs"


@pytest.mark.parametrize(
    ("var", "value", "fragment"),
    [
        ("OLLAMA_BASE_URL", "localhost:11434", "OLLAMA_BASE_URL"),
        ("CHUNK_SIZE", "abc", "integer"),
        ("CHUNK_SIZE", "100", "CHUNK_SIZE"),
        ("CHUNK_OVERLAP", "900", "CHUNK_OVERLAP"),
        ("TOP_K", "0", "TOP_K"),
        ("HYBRID_SEARCH", "maybe", "true/false"),
        ("MAX_UPLOAD_MB", "0", "MAX_UPLOAD_MB"),
        ("LOG_LEVEL", "LOUD", "LOG_LEVEL"),
    ],
)
def test_invalid_values_raise_actionable_errors(monkeypatch, clean_env, var, value, fragment):
    monkeypatch.setenv(var, value)
    with pytest.raises(ConfigError, match=fragment):
        load_settings(clean_env)


def test_candidate_k_must_cover_top_k():
    with pytest.raises(ConfigError, match="CANDIDATE_K"):
        Settings(top_k=10, candidate_k=5).validate()


def test_with_overrides_validates(tmp_path: Path):
    s = Settings()
    assert s.with_overrides(index_dir=tmp_path).index_dir == tmp_path
    with pytest.raises(ConfigError):
        s.with_overrides(chunk_size=50)
