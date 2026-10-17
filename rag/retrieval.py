"""Chemical-aware hybrid retrieval.

1. Detect which indexed chemicals the question is about (names, aliases, CAS).
2. Restrict the search to those chemicals' SDS chunks. Questions naming
   several chemicals are searched per chemical so each gets evidence.
3. Rank candidates with dense (FAISS) and lexical (BM25) search and fuse the
   two rankings with Reciprocal Rank Fusion.
4. Optionally rerank the fused candidates with a cross-encoder.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np

from rag.bm25 import BM25
from rag.chemicals import ChemicalMention, detect_chemicals
from rag.chunking import Chunk
from rag.config import Settings

if TYPE_CHECKING:
    from rag.indexing import KnowledgeBase

log = logging.getLogger(__name__)


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class ScopeReason(str, Enum):
    QUESTION = "question"  # chemicals named in the question
    HISTORY = "history"  # follow-up question: chemicals carried over from earlier turns
    USER = "user"  # user picked a filter in the UI
    NONE = "none"  # no chemical identified: search everything


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    mention: ChemicalMention
    scope: tuple[str, ...] = ()
    scope_reason: ScopeReason = ScopeReason.NONE
    notes: list[str] = field(default_factory=list)


def rrf_fuse(rankings: list[list[int]], k: int) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked lists of ids."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    def __init__(
        self,
        kb: KnowledgeBase,
        embedder: QueryEmbedder,
        settings: Settings,
        reranker: Reranker | None = None,
        *,
        chemical_filtering: bool = True,
    ) -> None:
        self.kb = kb
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker
        # Only disabled for evaluation ablations.
        self.chemical_filtering = chemical_filtering
        self._bm25: BM25 | None = None
        self.rows_by_chemical: dict[str, list[int]] = {}
        for i, c in enumerate(kb.chunks):
            self.rows_by_chemical.setdefault(c.chemical_key, []).append(i)

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([c.search_text for c in self.kb.chunks])
        return self._bm25

    # ── public API ─────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        history_chemicals: tuple[str, ...] = (),
        scope_override: tuple[str, ...] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.settings.top_k
        mention = detect_chemicals(query, self.kb.identities)
        result = RetrievalResult(chunks=[], mention=mention)
        if not self.kb.chunks:
            result.notes.append("The knowledge base is empty.")
            return result

        scope: tuple[str, ...]
        reason: ScopeReason
        if not self.chemical_filtering:
            scope, reason = (), ScopeReason.NONE
        elif scope_override:
            scope, reason = tuple(k for k in scope_override if k in self.rows_by_chemical), ScopeReason.USER
            ignored = [k for k in mention.present if k not in scope]
            if ignored:
                result.notes.append(
                    "The question mentions " + ", ".join(ignored) + " but the active filter excludes it."
                )
        elif mention.present:
            scope, reason = mention.present, ScopeReason.QUESTION
        elif mention.missing:
            # Only chemicals we have no SDS for: do not guess with another chemical's data.
            result.scope_reason = ScopeReason.QUESTION
            return result
        elif history_chemicals:
            scope = tuple(k for k in history_chemicals if k in self.rows_by_chemical)
            reason = ScopeReason.HISTORY if scope else ScopeReason.NONE
        else:
            scope, reason = (), ScopeReason.NONE
        result.scope, result.scope_reason = scope, reason

        if not scope:
            result.chunks = self._search(query, None, top_k)
        elif len(scope) == 1:
            result.chunks = self._search(query, self.rows_by_chemical[scope[0]], top_k)
        else:
            per = max(2, math.ceil(top_k / len(scope)))
            groups = [self._search(query, self.rows_by_chemical[key], per) for key in scope]
            merged: list[RetrievedChunk] = []
            for i in range(per):
                merged.extend(g[i] for g in groups if i < len(g))
            result.chunks = merged
        log.debug("retrieve scope=%s reason=%s -> %d chunks", scope, reason.value, len(result.chunks))
        return result

    # ── internals ──────────────────────────────────────────────────────────
    def _dense(self, query: str, allowed: list[int] | None, k: int) -> list[int]:
        import faiss

        qv = self.embedder.embed_query(query)
        k = min(k, len(allowed) if allowed is not None else self.kb.index.ntotal)
        if k <= 0:
            return []
        if allowed is None:
            _, ids = self.kb.index.search(qv, k)
        else:
            sel = faiss.IDSelectorBatch(np.asarray(allowed, dtype="int64"))
            _, ids = self.kb.index.search(qv, k, params=faiss.SearchParameters(sel=sel))
        return [int(i) for i in ids[0] if i >= 0]

    def _search(self, query: str, allowed: list[int] | None, k: int) -> list[RetrievedChunk]:
        cand_k = max(self.settings.candidate_k, k)
        dense = self._dense(query, allowed, cand_k)
        dense_rank = {cid: r for r, cid in enumerate(dense, start=1)}
        if self.settings.hybrid_search:
            sparse = [cid for cid, _ in self.bm25.top(query, cand_k, allowed)]
            sparse_rank = {cid: r for r, cid in enumerate(sparse, start=1)}
            fused = rrf_fuse([dense, sparse], self.settings.rrf_k)
        else:
            sparse_rank = {}
            fused = rrf_fuse([dense], self.settings.rrf_k)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:cand_k]

        if self.reranker is not None and ranked:
            texts = [self.kb.chunks[cid].search_text for cid, _ in ranked]
            try:
                rscores = self.reranker.score(query, texts)
                ranked = sorted(zip((cid for cid, _ in ranked), rscores, strict=True),
                                key=lambda kv: -kv[1])
            except Exception as exc:  # never let an optional component break answering
                log.warning("Reranker failed, using fused ranking: %s", exc)

        return [
            RetrievedChunk(
                chunk=self.kb.chunks[cid],
                score=float(score),
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
            )
            for cid, score in ranked[:k]
        ]
