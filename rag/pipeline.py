"""End-to-end question answering: retrieve -> build prompt -> generate -> validate citations."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from rag.citations import Source, build_sources, check_citations, format_context, render_citations, strip_citations
from rag.config import Settings
from rag.indexing import KnowledgeBase, load_knowledge_base
from rag.llm import Message, OllamaClient
from rag.prompts import REFUSAL, SYSTEM_PROMPT, USER_TEMPLATE, is_refusal, missing_note, no_sds_answer
from rag.retrieval import QueryEmbedder, Reranker, RetrievalResult, Retriever, ScopeReason

log = logging.getLogger(__name__)


class ChatModel(Protocol):
    def stream_chat(self, messages: list[Message]) -> Iterator[str]: ...


@dataclass(frozen=True)
class Turn:
    """One message of conversation history."""

    role: str  # "user" | "assistant"
    content: str
    chemicals: tuple[str, ...] = ()  # retrieval scope used for this user turn


@dataclass
class PreparedQuery:
    question: str
    retrieval: RetrievalResult
    sources: list[Source]
    messages: list[Message]
    canned_answer: str | None = None  # answer without calling the LLM

    @property
    def scope(self) -> tuple[str, ...]:
        return self.retrieval.scope


@dataclass
class Answer:
    text: str  # raw model output (with [S#] labels)
    markdown: str  # labels rendered as [1], [2], ...
    sources: list[Source]  # everything supplied to the model
    cited_sources: list[Source]  # in marker order
    cited_labels: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    unlisted_files: list[str] = field(default_factory=list)
    refused: bool = False
    uncited: bool = False
    interrupted: bool = False
    error: str | None = None

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.interrupted:
            out.append("Generation stopped before the answer was complete.")
        if self.invalid_labels:
            out.append(
                "The model referred to sources that were not provided ("
                + ", ".join(self.invalid_labels) + "); those references were removed."
            )
        if self.unlisted_files:
            out.append(
                "The answer mentions document names that were not retrieved: "
                + ", ".join(self.unlisted_files) + ". Treat those statements as unverified."
            )
        if self.uncited:
            out.append("The model did not cite any source for this answer. Verify it against the excerpts below.")
        return out


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        kb: KnowledgeBase,
        embedder: QueryEmbedder,
        llm: ChatModel,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.kb = kb
        self.llm = llm
        self.retriever = Retriever(kb, embedder, settings, reranker)
        self._names = {i.key: i.name for i in kb.identities}

    @classmethod
    def from_settings(cls, settings: Settings) -> RagPipeline:
        from rag.embeddings import Embedder

        embedder = Embedder(settings.embedding_model, settings.embedding_device)
        kb = load_knowledge_base(settings)
        reranker = None
        if settings.reranker_enabled:
            from rag.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(settings.reranker_model, settings.embedding_device)
        return cls(settings, kb, embedder, make_llm(settings), reranker)

    def display_name(self, key: str) -> str:
        return self._names.get(key, key[:1].upper() + key[1:])

    # ── step 1: retrieval + prompt ────────────────────────────────────────
    def prepare(
        self,
        question: str,
        history: list[Turn],
        scope_override: tuple[str, ...] | None = None,
    ) -> PreparedQuery:
        question = question.strip()[: self.settings.max_question_chars]
        history_chems: tuple[str, ...] = ()
        for turn in reversed(history):
            if turn.role == "user" and turn.chemicals:
                history_chems = turn.chemicals
                break

        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(
            question, history_chemicals=history_chems, scope_override=scope_override
        )
        log.info(
            "retrieval: %d chunks, scope=%s (%s), missing=%s, %.0f ms",
            len(retrieval.chunks), retrieval.scope, retrieval.scope_reason.value,
            retrieval.mention.missing, (time.perf_counter() - t0) * 1000,
        )
        sources = build_sources(retrieval.chunks)
        missing = [self.display_name(k) for k in retrieval.mention.missing]

        canned = None
        if not sources:
            if missing and not retrieval.mention.present:
                canned = no_sds_answer(missing)
            elif not self.kb.chunks:
                canned = "The knowledge base is empty. Add SDS PDFs to the documents folder or upload them."
            else:
                canned = REFUSAL

        notes = ""
        if missing and retrieval.mention.present:
            notes += missing_note(missing)
        if retrieval.scope_reason is ScopeReason.HISTORY and retrieval.scope:
            names = ", ".join(self.display_name(k) for k in retrieval.scope)
            notes += f"Note: this is a follow-up question about {names}.\n\n"
        for n in retrieval.notes:
            notes += f"Note: {n}\n\n"

        messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
        window = history[-2 * self.settings.max_history_turns :] if self.settings.max_history_turns else []
        for turn in window:
            content = strip_citations(turn.content) if turn.role == "assistant" else turn.content
            messages.append({"role": turn.role, "content": content})
        messages.append({
            "role": "user",
            "content": USER_TEMPLATE.format(context=format_context(sources), notes=notes, question=question),
        })
        return PreparedQuery(question, retrieval, sources, messages, canned)

    # ── step 2: generation ─────────────────────────────────────────────────
    def stream(self, prepared: PreparedQuery) -> Iterator[str]:
        if prepared.canned_answer is not None:
            yield prepared.canned_answer
            return
        yield from self.llm.stream_chat(prepared.messages)

    # ── step 3: validation ─────────────────────────────────────────────────
    def finalize(self, prepared: PreparedQuery, text: str, *, interrupted: bool = False,
                 error: str | None = None) -> Answer:
        text = text.strip()
        if prepared.canned_answer is not None:
            return Answer(text=text, markdown=text, sources=[], cited_sources=[], refused=True)
        check = check_citations(text, prepared.sources)
        markdown, cited = render_citations(text, prepared.sources)
        refused = is_refusal(text) and not check.cited_labels
        return Answer(
            text=text,
            markdown=markdown,
            sources=prepared.sources,
            cited_sources=cited,
            cited_labels=check.cited_labels,
            invalid_labels=check.invalid_labels,
            unlisted_files=check.unlisted_files,
            refused=refused,
            uncited=bool(text) and not refused and not check.cited_labels,
            interrupted=interrupted,
            error=error,
        )

    def complete(self, prepared: PreparedQuery) -> Answer:
        return self.finalize(prepared, "".join(self.stream(prepared)))

    def ask(self, question: str, history: list[Turn] | None = None) -> Answer:
        return self.complete(self.prepare(question, history or []))


def make_llm(settings: Settings) -> OllamaClient:
    return OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
        connect_timeout_s=settings.llm_connect_timeout_s,
        read_timeout_s=settings.llm_read_timeout_s,
    )
