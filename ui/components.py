"""Rendering helpers for the Streamlit UI (no RAG logic here)."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from rag.chemicals import ChemicalIdentity
from rag.citations import Source

CSS_PATH = Path(__file__).with_name("style.css")


def inject_css() -> None:
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def escape_markdown(text: str) -> str:
    """Render untrusted text (PDF content) literally in st.markdown."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def sanitize_answer(text: str) -> str:
    """Neutralise markdown that could load remote content or mislead.

    Model output is influenced by document text, so images are dropped,
    links are shown as plain text with their target, and dollar signs are
    escaped so values are not rendered as LaTeX.
    """
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("$", r"\$")


def header() -> None:
    st.html('<div class="sds-eyebrow">Chemical safety · Safety Data Sheets</div>')
    st.markdown("# Ask your Safety Data Sheets")
    st.html(
        '<p class="sds-lede">Answers about hazards, protective equipment, first aid, storage and '
        "physical properties, drawn only from the SDS documents in the knowledge base, with "
        "every statement linked to its page.</p>"
    )


def disclaimer() -> None:
    st.html(
        '<div class="sds-notice" role="note"><b>Not a substitute for the official SDS.</b> '
        "Answers can be incomplete or out of date. Check the supplier's current Safety Data Sheet "
        "and follow your site procedures. In an emergency, call your local emergency number or "
        "poison control centre.</div>"
    )


def source_block(index: int, src: Source) -> None:
    meta = f"{src.filename} · {src.page_label} · {src.section_label}"
    st.html(
        f'<div class="sds-src-head"><span class="sds-src-num">{index}</span>'
        f"<b>{html.escape(src.chemical_name)}</b>"
        f'<span class="sds-src-meta">{html.escape(meta)}</span></div>'
        f'<div class="sds-src-text">{html.escape(src.text)}</div>'
    )


def sources_panel(cited: Sequence[Source], retrieved: Sequence[Source]) -> None:
    if not retrieved:
        return
    cited_labels = {s.label for s in cited}
    extra = [s for s in retrieved if s.label not in cited_labels]
    if cited:
        title = f"Sources · {len(cited)} cited"
        if extra:
            title += f", {len(extra)} more retrieved"
    else:
        title = f"Retrieved excerpts · {len(retrieved)}"
    with st.expander(title, icon=":material/description:"):
        for i, src in enumerate(cited, start=1):
            source_block(i, src)
        if extra:
            if cited:
                st.caption("Also retrieved but not cited")
            for offset, src in enumerate(extra, start=len(cited) + 1):
                source_block(offset, src)


def document_list(docs: Sequence[tuple[ChemicalIdentity, str, int]]) -> None:
    """docs: (identity, filename, pages)."""
    rows = []
    for ident, filename, pages in docs:
        cas = f"CAS {ident.cas}" if ident.cas else "CAS not found"
        rows.append(
            f'<div class="sds-doc"><b>{html.escape(ident.name)}</b> · {html.escape(cas)}'
            f'<span class="sds-src-meta">{html.escape(filename)} · {pages} pages</span></div>'
        )
    st.html("".join(rows))


def example_questions(identities: Sequence[ChemicalIdentity]) -> list[str]:
    """Example prompts built from the chemicals actually indexed."""
    names = [i.name for i in identities]
    if not names:
        return []
    templates = [
        "What is the flash point of {0}?",
        "What protective equipment is recommended when handling {0}?",
        "What first-aid measures apply after eye contact with {0}?",
        "How should {0} be stored?",
        "What are the hazard statements for {0}?",
        "Compare the flash points of {0} and {1}.",
    ]
    out = []
    for i, tpl in enumerate(templates):
        a = names[i % len(names)]
        if "{1}" in tpl:
            if len(names) < 2:
                continue
            b = names[(i + 1) % len(names)]
            out.append(tpl.format(a, b))
        else:
            out.append(tpl.format(a))
    return out
