"""Chemical Safety RAG Assistant: Streamlit UI.

All retrieval, prompting and citation logic lives in ``rag/``; this module
only manages UI state and rendering.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import streamlit as st

from rag.config import Settings, configure_logging, load_settings
from rag.errors import ConfigError, IndexBusyError, KnowledgeBaseError, LLMError, ModelNotFoundError
from rag.export import ExportMessage, conversation_to_markdown
from rag.health import HealthState, HealthStatus, check_ollama
from rag.indexing import (
    KBState,
    KBStatus,
    KnowledgeBase,
    SyncReport,
    assess_status,
    load_knowledge_base,
    read_current_version,
    sync_knowledge_base,
)
from rag.pipeline import PreparedQuery, RagPipeline, Turn, make_llm
from rag.uploads import UploadStatus, stage_uploads
from ui import components as ui

log = logging.getLogger("rag.app")

st.set_page_config(
    page_title="SDS Assistant · Chemical Safety",
    page_icon=":material/science:",
    layout="centered",
    initial_sidebar_state="auto",  # open on desktop, collapsed on phones
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources (shared across sessions)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return load_settings()


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder(model: str, device: str) -> Any:
    from rag.embeddings import Embedder

    return Embedder(model, device)


@st.cache_resource(show_spinner=False)
def get_reranker(model: str, device: str) -> Any:
    from rag.rerank import CrossEncoderReranker

    return CrossEncoderReranker(model, device)


@st.cache_resource(show_spinner="Loading knowledge base…", max_entries=2)
def get_kb(version: str) -> KnowledgeBase:
    return load_knowledge_base(get_settings(), version)


@st.cache_resource(show_spinner=False, max_entries=2)
def get_pipeline(version: str) -> RagPipeline:
    s = get_settings()
    reranker = get_reranker(s.reranker_model, s.embedding_device) if s.reranker_enabled else None
    return RagPipeline(s, get_kb(version), get_embedder(s.embedding_model, s.embedding_device),
                       make_llm(s), reranker)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],  # list of dicts, see add_assistant_message()
        "health": None,
        "uploader_key": 0,
        "upload_results": [],
        "scope_filter": [],
        "build_attempted": False,
        "sync_notes": [],  # (kind, text) from the last index sync
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def get_health(settings: Settings, force: bool = False) -> HealthStatus:
    h: HealthStatus | None = st.session_state.health
    max_age = 60 if (h and h.ok) else 10
    if force or h is None or time.time() - h.checked_at > max_age:
        h = check_ollama(settings.ollama_base_url, settings.ollama_model, settings.health_timeout_s)
        st.session_state.health = h
    return h


def history_turns() -> list[Turn]:
    turns: list[Turn] = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            turns.append(Turn("user", m["content"], tuple(m.get("scope", ()))))
        elif not m.get("error"):
            turns.append(Turn("assistant", m.get("raw", m["content"])))
    return turns


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge-base lifecycle
# ─────────────────────────────────────────────────────────────────────────────
def run_sync(settings: Settings, *, full: bool, label: str) -> SyncReport | None:
    embedder = get_embedder(settings.embedding_model, settings.embedding_device)
    bar = st.progress(0.0, text=label)

    def progress(stage: str, done: int, total: int) -> None:
        names = {"scan": "Scanning documents", "parse": "Reading PDFs", "embed": "Embedding text",
                 "write": "Saving index"}
        weights = {"scan": (0.0, 0.05), "parse": (0.05, 0.3), "embed": (0.3, 0.95), "write": (0.95, 1.0)}
        lo, hi = weights.get(stage, (0.0, 1.0))
        frac = lo + (hi - lo) * (done / total if total else 1.0)
        bar.progress(min(frac, 1.0), text=f"{names.get(stage, stage)} ({done}/{total})")

    try:
        report = sync_knowledge_base(settings, embedder, full_rebuild=full, progress=progress)
    except IndexBusyError as exc:
        bar.empty()
        st.warning(str(exc))
        return None
    except Exception:
        bar.empty()
        log.exception("Index sync failed")
        st.error("Building the index failed. The previous index (if any) is unchanged. See the server log.")
        return None
    bar.empty()
    # Callers rerun the script right after a sync, so notes are kept in state and shown in the sidebar.
    st.session_state.sync_notes = [("warning", f"Skipped {src}: {msg}") for src, msg in report.failed] + [
        ("info", f"{src} has the same content as {original}, so it was not indexed twice.")
        for src, original in report.duplicates
    ]
    st.toast(f"Knowledge base ready: {report.summary()}", icon=":material/check_circle:")
    return report


def load_active_kb(settings: Settings) -> tuple[KnowledgeBase | None, KBStatus]:
    kb, err = None, None
    try:
        version = read_current_version(settings.index_dir)
        if version:
            kb = get_kb(version)
        else:
            err = KnowledgeBaseError("No index has been built yet.")
    except KnowledgeBaseError as exc:
        err = exc
    return kb, assess_status(settings, kb, err)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def sidebar(
    settings: Settings, kb: KnowledgeBase | None, status: KBStatus, health: HealthStatus
) -> Any:
    """Render the sidebar; returns the (still empty) container for the conversation controls."""
    with st.sidebar:
        st.markdown("### SDS Assistant")
        st.caption("Local, offline question answering over chemical Safety Data Sheets.")

        # Model status
        st.markdown("**Language model**")
        if health.ok:
            st.badge(f"{settings.ollama_model} · connected", icon=":material/check_circle:", color="green")
        else:
            label = {HealthState.MODEL_MISSING: "model not installed", HealthState.TIMEOUT: "not responding",
                     HealthState.SERVER_DOWN: "offline"}.get(health.state, "error")
            st.badge(f"Ollama · {label}", icon=":material/error:", color="red")
            st.caption(health.message)
            if st.button("Check again", icon=":material/refresh:", key="retry_health"):
                get_health(settings, force=True)
                st.rerun()

        st.divider()

        # Knowledge base
        st.markdown("**Knowledge base**")
        if kb is not None:
            docs = kb.manifest.documents
            st.caption(f"{len(docs)} documents · {len(kb.chunks)} indexed passages · built "
                       f"{kb.manifest.created_at.replace('T', ' ')[:16]} UTC")
            if docs:
                ui.document_list([(d.chemical, d.filename, d.pages) for d in docs])
            for kind, text in st.session_state.sync_notes:
                getattr(st, kind)(text)
            if status.state is KBState.STALE:
                st.warning(status.message, icon=":material/sync_problem:")
                if st.button("Update index", icon=":material/sync:", key="sync_btn") and run_sync(
                    settings, full=False, label="Updating index…"
                ):
                    st.rerun()
            identities = kb.identities
            names = {i.key: i.name for i in identities}
            # Drop selections for chemicals that are no longer indexed.
            st.session_state.scope_filter = [k for k in st.session_state.scope_filter if k in names]
            if len(identities) > 1:
                st.multiselect(
                    "Limit answers to",
                    options=list(names),
                    format_func=lambda k: names.get(k, k),
                    key="scope_filter",
                    placeholder="All chemicals (detected from your question)",
                    help="By default the assistant detects which chemicals your question is about. "
                    "Choose chemicals here to restrict every answer to their SDS.",
                )

        # Uploads
        with st.expander("Add SDS documents", icon=":material/upload_file:"):
            files = st.file_uploader(
                "PDF Safety Data Sheets",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.uploader_key}",
                help=f"Text-based PDFs up to {settings.max_upload_mb} MB each. Scanned images are not supported.",
            )
            disabled = not files or (kb is None and status.state is not KBState.MISSING)
            if st.button("Add to knowledge base", type="primary", disabled=disabled, icon=":material/add:"):
                handle_uploads(settings, kb, files or [])
            for kind, text in st.session_state.upload_results:
                getattr(st, kind)(text)

        st.divider()

        # Conversation controls are filled in at the end of the run, after any new answer
        # has been added; rendered here they would export a conversation missing its last turn.
        conversation_slot = st.container()
        st.caption(f"Embeddings: {settings.embedding_model.split('/')[-1]} · Retrieval: "
                   f"{'hybrid (vector + keyword)' if settings.hybrid_search else 'vector'}"
                   f"{' + reranker' if settings.reranker_enabled else ''}")
    return conversation_slot


def conversation_controls(slot: Any, settings: Settings, kb: KnowledgeBase | None) -> None:
    with slot:
        st.markdown("**Conversation**")
        msgs = st.session_state.messages
        if st.button("Clear conversation", icon=":material/delete_sweep:", disabled=not msgs,
                     help="Clears this conversation only. The knowledge base is not affected.",
                     use_container_width=True):
            st.session_state.messages = []
            st.toast("Conversation cleared")
            st.rerun()
        st.download_button(
            "Export as Markdown",
            data=export_markdown(settings, kb) if msgs else "",
            file_name=f"sds-conversation-{datetime.now():%Y%m%d-%H%M}.md",
            mime="text/markdown",
            icon=":material/download:",
            disabled=not msgs,
            help="Download the conversation with its citations.",
            use_container_width=True,
        )
        st.divider()


def handle_uploads(settings: Settings, kb: KnowledgeBase | None, files: list[Any]) -> None:
    results: list[tuple[str, str]] = []
    payload = [(f.name, f.getvalue()) for f in files]
    staged = stage_uploads(settings, payload, kb.manifest if kb else None)
    for o in staged.outcomes:
        if o.status is UploadStatus.REJECTED:
            results.append(("error", f"{o.filename}: {o.message}"))
        elif o.status is UploadStatus.DUPLICATE:
            results.append(("info", f"{o.filename}: {o.message}"))
    if staged.saved:
        report = run_sync(settings, full=False, label=f"Indexing {len(staged.saved)} document(s)…")
        if report is None:
            # Nothing was written (sync failed or another writer was busy): undo every upload.
            staged.rollback()
            results.append(("error", "Indexing failed; the uploaded files were not added."))
        else:
            kept, rolled_back = staged.settle(settings.docs_dir, dict(report.failed))
            for o in kept:
                verb = "replaced and indexed" if o.status is UploadStatus.REPLACED else "added and indexed"
                results.append(("success", f"{o.filename}: {verb}."))
            for o in rolled_back:
                results.append(("error", f"{o.filename}: could not be indexed, not added. {o.message}"))
            if rolled_back:
                # Removing/restoring files changed the folder again; bring the index back in line.
                run_sync(settings, full=False, label="Restoring previous documents…")
    st.session_state.upload_results = results
    st.session_state.uploader_key += 1
    st.rerun()


def export_markdown(settings: Settings, kb: KnowledgeBase | None) -> str:
    msgs = [
        ExportMessage(m["role"], m["content"], m.get("sources", []), m.get("warnings", []))
        for m in st.session_state.messages
    ]
    return conversation_to_markdown(msgs, model=settings.ollama_model,
                                    index_version=kb.version if kb else "none")


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────
def render_message(m: dict[str, Any]) -> None:
    avatar = ":material/person:" if m["role"] == "user" else ":material/science:"
    with st.chat_message(m["role"], avatar=avatar):
        if m["role"] == "user":
            st.markdown(ui.escape_markdown(m["content"]))
            return
        if m.get("error"):
            st.error(m["content"], icon=":material/error:")
            return
        st.markdown(ui.sanitize_answer(m["content"]))
        for w in m.get("warnings", []):
            st.caption(f":material/warning: {w}")
        ui.sources_panel(m.get("sources", []), m.get("retrieved", []))


def stream_answer(pipeline: RagPipeline, prepared: PreparedQuery, sink: list[str],
                  failure: dict[str, Exception]) -> Iterator[str]:
    try:
        for piece in pipeline.stream(prepared):
            sink.append(piece)
            yield piece
    except LLMError as exc:
        failure["error"] = exc


def answer_question(pipeline: RagPipeline, question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])

    with st.chat_message("assistant", avatar=":material/science:"):
        scope_override = tuple(st.session_state.scope_filter) or None
        prepared = pipeline.prepare(question, history_turns()[:-1], scope_override)
        st.session_state.messages[-1]["scope"] = list(prepared.scope)
        if prepared.scope:
            st.caption("Searching: " + ", ".join(pipeline.display_name(k) for k in prepared.scope))

        sink: list[str] = []
        failure: dict[str, Exception] = {}
        placeholder = st.empty()
        t0 = time.perf_counter()
        with placeholder.container(), st.spinner("Reading the Safety Data Sheets…"):
            st.write_stream(stream_answer(pipeline, prepared, sink, failure))
        placeholder.empty()
        elapsed = time.perf_counter() - t0

        err = failure.get("error")
        text = "".join(sink)
        if err is not None:
            log.warning("Generation failed after %d chars: %s", len(text), err)
            if not isinstance(err, ModelNotFoundError):
                st.session_state.health = None  # force a fresh health check
            if not text.strip():
                failed_msg = {"role": "assistant", "content": f"Could not generate an answer. {err}", "error": True}
                st.session_state.messages.append(failed_msg)
                st.error(failed_msg["content"], icon=":material/error:")
                return

        answer = pipeline.finalize(prepared, text, interrupted=err is not None,
                                   error=str(err) if err else None)
        log.info("answer: %.1fs, %d chars, cited=%s, refused=%s, warnings=%d",
                 elapsed, len(answer.text), answer.cited_labels, answer.refused, len(answer.warnings))
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": answer.markdown,
            "raw": answer.text,
            "sources": answer.cited_sources,
            "retrieved": answer.sources,
            "warnings": answer.warnings,
        }
        st.session_state.messages.append(msg)
        st.markdown(ui.sanitize_answer(msg["content"]))
        for w in msg["warnings"]:
            st.caption(f":material/warning: {w}")
        ui.sources_panel(msg["sources"], msg["retrieved"])


def empty_state(kb: KnowledgeBase) -> str | None:
    """Intro for first-time users; returns a question if an example was clicked."""
    if not kb.chunks:
        st.info("The knowledge base is empty. Add SDS PDFs from the sidebar to get started.",
                icon=":material/upload_file:")
        return None
    st.markdown("##### Try asking")
    picked = None
    examples = ui.example_questions(kb.identities)
    with st.container(key="examples"):
        cols = st.columns(2)
        for i, q in enumerate(examples):
            if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                picked = q
    st.caption(
        "Each answer cites numbered sources. Open **Sources** under an answer to read the exact SDS "
        "passage and page it came from."
    )
    return picked


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    try:
        settings = get_settings()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.caption("Fix the value in your environment or .env file and restart the app.")
        st.stop()
    configure_logging(settings.log_level)
    ui.inject_css()
    init_state()

    health = get_health(settings)
    kb, status = load_active_kb(settings)

    ui.header()
    ui.disclaimer()

    # First run: build the index automatically from the bundled documents.
    if status.state is KBState.MISSING and not st.session_state.build_attempted:
        st.session_state.build_attempted = True
        st.info("Building the knowledge base from the documents folder for the first time…",
                icon=":material/database:")
        if run_sync(settings, full=False, label="Building knowledge base…"):
            st.rerun()
    if status.state in (KBState.INCOMPATIBLE, KBState.CORRUPT, KBState.MISSING):
        st.error(status.message, icon=":material/database:")
        if st.button("Rebuild index", type="primary", icon=":material/build:") and run_sync(
            settings, full=True, label="Rebuilding index…"
        ):
            st.rerun()

    conversation_slot = sidebar(settings, kb, status, health)

    if kb is None:
        conversation_controls(conversation_slot, settings, kb)
        return

    pipeline = get_pipeline(kb.version)
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "user":
        # Every completed or failed answer appends an assistant message, so a trailing user
        # message means the script was stopped mid-answer (e.g. a new question sent while streaming).
        msgs.append({"role": "assistant", "error": True,
                     "content": "This answer was interrupted before it finished. Ask again for a complete answer."})
    picked = None
    if not msgs:
        picked = empty_state(kb)
    for m in st.session_state.messages:
        render_message(m)

    if not health.ok:
        placeholder_text = "Waiting for the language model. See the status in the sidebar."
    elif not kb.chunks:
        placeholder_text = "Add SDS documents to start asking questions."
    else:
        placeholder_text = "Ask about hazards, PPE, first aid, storage, properties…"
    typed = st.chat_input(placeholder_text, disabled=not (health.ok and kb.chunks),
                          max_chars=settings.max_question_chars)
    question = (picked or typed or "").strip()
    if question:
        if not get_health(settings, force=True).ok:
            st.warning("The language model is not available right now. Check the sidebar and try again.")
        else:
            answer_question(pipeline, question)
    conversation_controls(conversation_slot, settings, kb)


main()
