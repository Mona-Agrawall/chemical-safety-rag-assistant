"""Centralised, environment-driven configuration.

Values are read from environment variables (optionally loaded from a
``.env`` file) and validated once. Every setting has a sensible default so
the app runs with zero configuration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

from rag.errors import ConfigError

BASE_DIR = Path(__file__).resolve().parent.parent

# Bump when chunking/metadata logic changes in a way that requires re-indexing.
PIPELINE_VERSION = "sds-section-v1"
INDEX_FORMAT_VERSION = 2


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true/false, got {raw!r}")


def _path(name: str, default: Path) -> Path:
    p = Path(_env(name, str(default))).expanduser()
    return p if p.is_absolute() else (BASE_DIR / p)


@dataclass(frozen=True)
class Settings:
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 8192
    llm_connect_timeout_s: float = 5.0
    llm_read_timeout_s: float = 180.0
    health_timeout_s: float = 3.0

    # Embeddings / retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 6
    candidate_k: int = 30
    hybrid_search: bool = True
    rrf_k: int = 60
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_history_turns: int = 4

    # Storage
    docs_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    index_dir: Path = field(default_factory=lambda: BASE_DIR / "storage" / "index")
    uploads_subdir: str = "uploads"

    # Limits
    max_upload_mb: int = 20
    max_pdf_pages: int = 200
    max_question_chars: int = 2000

    log_level: str = "INFO"

    @property
    def uploads_dir(self) -> Path:
        return self.docs_dir / self.uploads_subdir

    def with_overrides(self, **kwargs: object) -> Settings:
        new = replace(self, **kwargs)  # type: ignore[arg-type]
        new.validate()
        return new

    def validate(self) -> None:
        url = urlparse(self.ollama_base_url)
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ConfigError(
                f"OLLAMA_BASE_URL must look like http://host:port, got {self.ollama_base_url!r}"
            )
        if not self.ollama_model:
            raise ConfigError("OLLAMA_MODEL must not be empty")
        if not self.embedding_model:
            raise ConfigError("EMBEDDING_MODEL must not be empty")
        if self.chunk_size < 200:
            raise ConfigError("CHUNK_SIZE must be at least 200 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size // 2:
            raise ConfigError("CHUNK_OVERLAP must be >= 0 and less than half of CHUNK_SIZE")
        if not 1 <= self.top_k <= 20:
            raise ConfigError("TOP_K must be between 1 and 20")
        if self.candidate_k < self.top_k:
            raise ConfigError("CANDIDATE_K must be >= TOP_K")
        if self.rrf_k < 1:
            raise ConfigError("RRF_K must be positive")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ConfigError("LLM_TEMPERATURE must be between 0 and 2")
        if self.llm_num_ctx < 2048:
            raise ConfigError("LLM_NUM_CTX must be at least 2048")
        for name in ("llm_connect_timeout_s", "llm_read_timeout_s", "health_timeout_s"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name.upper()} must be positive")
        if not 1 <= self.max_upload_mb <= 500:
            raise ConfigError("MAX_UPLOAD_MB must be between 1 and 500")
        if self.max_pdf_pages < 1:
            raise ConfigError("MAX_PDF_PAGES must be positive")
        if self.max_history_turns < 0:
            raise ConfigError("MAX_HISTORY_TURNS must be >= 0")
        if self.log_level.upper() not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {self.log_level!r} is not a valid logging level")
        if "/" in self.uploads_subdir or "\\" in self.uploads_subdir or ".." in self.uploads_subdir:
            raise ConfigError("UPLOADS_SUBDIR must be a single directory name")


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from the environment (and ``.env`` if present) and validate."""
    from dotenv import load_dotenv

    # Real environment variables take precedence over the .env file.
    load_dotenv(env_file or (BASE_DIR / ".env"), override=False)

    s = Settings(
        ollama_base_url=_env("OLLAMA_BASE_URL", Settings.ollama_base_url).rstrip("/"),
        ollama_model=_env("OLLAMA_MODEL", Settings.ollama_model),
        llm_temperature=_float("LLM_TEMPERATURE", Settings.llm_temperature),
        llm_num_ctx=_int("LLM_NUM_CTX", Settings.llm_num_ctx),
        llm_connect_timeout_s=_float("LLM_CONNECT_TIMEOUT_S", Settings.llm_connect_timeout_s),
        llm_read_timeout_s=_float("LLM_READ_TIMEOUT_S", Settings.llm_read_timeout_s),
        health_timeout_s=_float("HEALTH_TIMEOUT_S", Settings.health_timeout_s),
        embedding_model=_env("EMBEDDING_MODEL", Settings.embedding_model),
        embedding_device=_env("EMBEDDING_DEVICE", Settings.embedding_device),
        chunk_size=_int("CHUNK_SIZE", Settings.chunk_size),
        chunk_overlap=_int("CHUNK_OVERLAP", Settings.chunk_overlap),
        top_k=_int("TOP_K", Settings.top_k),
        candidate_k=_int("CANDIDATE_K", Settings.candidate_k),
        hybrid_search=_bool("HYBRID_SEARCH", Settings.hybrid_search),
        rrf_k=_int("RRF_K", Settings.rrf_k),
        reranker_enabled=_bool("RERANKER_ENABLED", Settings.reranker_enabled),
        reranker_model=_env("RERANKER_MODEL", Settings.reranker_model),
        max_history_turns=_int("MAX_HISTORY_TURNS", Settings.max_history_turns),
        docs_dir=_path("DOCS_DIR", BASE_DIR / "data"),
        index_dir=_path("INDEX_DIR", BASE_DIR / "storage" / "index"),
        uploads_subdir=_env("UPLOADS_SUBDIR", Settings.uploads_subdir),
        max_upload_mb=_int("MAX_UPLOAD_MB", Settings.max_upload_mb),
        max_pdf_pages=_int("MAX_PDF_PAGES", Settings.max_pdf_pages),
        max_question_chars=_int("MAX_QUESTION_CHARS", Settings.max_question_chars),
        log_level=_env("LOG_LEVEL", Settings.log_level).upper(),
    )
    s.validate()
    return s


def configure_logging(level: str) -> None:
    """Configure root logging once; safe to call on every Streamlit rerun."""
    root = logging.getLogger()
    if not any(getattr(h, "_rag_handler", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        handler._rag_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    logging.getLogger("rag").setLevel(level)
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
