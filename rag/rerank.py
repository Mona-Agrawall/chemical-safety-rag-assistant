"""Optional cross-encoder reranker (disabled by default, see README evaluation)."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        try:
            self._model = CrossEncoder(model_name, device=device, local_files_only=True)
        except Exception:
            log.info("Reranker model %s not cached locally; downloading once.", model_name)
            self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]
