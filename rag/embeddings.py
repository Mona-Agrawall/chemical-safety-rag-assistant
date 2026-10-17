"""Sentence-embedding model wrapper with model-specific query/passage prefixes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


def _prefixes(model_name: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) recommended by the model authors."""
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "bge" in name and "-en" in name:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


def model_revision(model_name: str) -> str:
    """Hugging Face snapshot hash of a cached model, or "" if unknown."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_name, local_files_only=True)).name
    except Exception:
        return ""


class Embedder:
    """Loads a sentence-transformers model and produces L2-normalised vectors."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix, self.passage_prefix = _prefixes(model_name)
        try:
            self._model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            log.info("Embedding model %s not cached locally; downloading once.", model_name)
            self._model = SentenceTransformer(model_name, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        if not dim:
            raise RuntimeError(f"Could not determine embedding dimension for {model_name}")
        self.dimension = int(dim)
        self.revision = model_revision(model_name)

    def embed_documents(
        self, texts: Sequence[str], batch_size: int = 32, progress: ProgressFn | None = None
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        out = []
        for start in range(0, len(texts), batch_size):
            batch = [self.passage_prefix + t for t in texts[start : start + batch_size]]
            vecs = self._model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
            out.append(np.asarray(vecs, dtype="float32"))
            if progress:
                progress(min(start + batch_size, len(texts)), len(texts))
        return np.vstack(out)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            [self.query_prefix + text], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vec, dtype="float32")
