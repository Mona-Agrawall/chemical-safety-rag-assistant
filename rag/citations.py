"""Source labelling, context formatting and citation validation.

Retrieved chunks are given short labels ([S1], [S2], ...) in the prompt. The
model cites those labels; it never writes file names or page numbers itself.
After generation every label is checked against the sources that were
actually supplied, so a citation can always be traced to a real document page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.retrieval import RetrievedChunk

LABEL_GROUP_RE = re.compile(r"\[\s*(S\s*\d+(?:\s*[,;&/]\s*(?:S\s*)?\d+)*)\s*\]", re.I)
_NUM_RE = re.compile(r"\d+")
PDF_NAME_RE = re.compile(r"(?<![\w\-.])[\w\-.]+\.pdf\b", re.I)


@dataclass(frozen=True)
class Source:
    label: str  # "S1"
    source: str
    chemical_name: str
    section: int
    section_title: str
    page_start: int
    page_end: int
    text: str
    score: float = 0.0

    @property
    def filename(self) -> str:
        return self.source.rsplit("/", 1)[-1]

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}–{self.page_end}"

    @property
    def section_label(self) -> str:
        return self.section_title if self.section == 0 else f"Section {self.section}: {self.section_title}"


def build_sources(retrieved: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            label=f"S{i}",
            source=r.chunk.source,
            chemical_name=r.chunk.chemical_name,
            section=r.chunk.section,
            section_title=r.chunk.section_title,
            page_start=r.chunk.page_start,
            page_end=r.chunk.page_end,
            text=r.chunk.text,
            score=r.score,
        )
        for i, r in enumerate(retrieved, start=1)
    ]


def _neutralise(text: str) -> str:
    """Stop document text from spoofing labels or closing the sources block."""
    text = re.sub(r"\[(\s*S\s*\d)", r"(\1", text, flags=re.I)
    return re.sub(r"</?\s*sources?\s*>", "", text, flags=re.I)


def format_context(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        blocks.append(
            f"[{s.label}] Chemical: {s.chemical_name} | File: {s.filename} | {s.page_label} | {s.section_label}\n"
            f"{_neutralise(s.text)}"
        )
    return "\n\n".join(blocks)


@dataclass
class CitationCheck:
    cited_labels: list[str] = field(default_factory=list)  # valid, in order of first use
    invalid_labels: list[str] = field(default_factory=list)  # cited but never supplied
    unlisted_files: list[str] = field(default_factory=list)  # file names written that were not supplied


def check_citations(text: str, sources: list[Source]) -> CitationCheck:
    valid = {s.label for s in sources}
    check = CitationCheck()
    for m in LABEL_GROUP_RE.finditer(text):
        for num in _NUM_RE.findall(m.group(1)):
            label = f"S{int(num)}"
            target = check.cited_labels if label in valid else check.invalid_labels
            if label not in target:
                target.append(label)
    supplied = {s.filename.lower() for s in sources}
    for name in PDF_NAME_RE.findall(text):
        name = name.strip().rsplit("/", 1)[-1]
        if name.lower() not in supplied and name not in check.unlisted_files:
            check.unlisted_files.append(name)
    return check


def render_citations(text: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """Replace [S#] labels with sequential [1], [2]... markers.

    Returns the rendered text and the cited sources in marker order. Labels
    that do not correspond to a supplied source are removed.
    """
    by_label = {s.label: s for s in sources}
    order: list[str] = []

    def repl(m: re.Match[str]) -> str:
        marks = []
        for num in _NUM_RE.findall(m.group(1)):
            label = f"S{int(num)}"
            if label not in by_label:
                continue
            if label not in order:
                order.append(label)
            mark = f"[{order.index(label) + 1}]"
            if mark not in marks:
                marks.append(mark)
        return "".join(marks)

    rendered = LABEL_GROUP_RE.sub(repl, text)
    rendered = re.sub(r"[ \t]+([.,;:])", r"\1", rendered)
    return rendered, [by_label[label] for label in order]


def strip_citations(text: str) -> str:
    """Remove citation markers (used when replaying history to the model)."""
    text = LABEL_GROUP_RE.sub("", text)
    return re.sub(r"\[\d+\]", "", text)
