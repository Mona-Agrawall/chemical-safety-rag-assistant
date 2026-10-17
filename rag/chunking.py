"""SDS-aware document segmentation.

Safety Data Sheets follow a 16-section structure (GHS / REACH Annex II).
Chunks are cut along those section boundaries, long sections are subdivided,
and every chunk records its chemical, section and the exact page range it
came from so citations can be traced back to the PDF.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

from rag.chemicals import ChemicalIdentity

SECTION_TITLES: dict[int, str] = {
    0: "Document header",
    1: "Identification",
    2: "Hazards identification",
    3: "Composition / information on ingredients",
    4: "First-aid measures",
    5: "Fire-fighting measures",
    6: "Accidental release measures",
    7: "Handling and storage",
    8: "Exposure controls / personal protection",
    9: "Physical and chemical properties",
    10: "Stability and reactivity",
    11: "Toxicological information",
    12: "Ecological information",
    13: "Disposal considerations",
    14: "Transport information",
    15: "Regulatory information",
    16: "Other information",
}
# The first word(s) a genuine heading must start with; rejects cross-references
# such as "see section 8 for details" or "Incompatible materials: see section 10. Keep away".
_SECTION_KEYWORDS: dict[int, tuple[str, ...]] = {
    1: ("identification", "product and company", "chemical product"),
    2: ("hazard",),
    3: ("composition", "information on ingredients"),
    4: ("first",),
    5: ("fire",),
    6: ("accidental",),
    7: ("handling",),
    8: ("exposure", "personal protection"),
    9: ("physical",),
    10: ("stability", "reactivity"),
    11: ("toxicolog",),
    12: ("ecolog",),
    13: ("disposal",),
    14: ("transport",),
    15: ("regulatory",),
    16: ("other information",),
}
_HEADING_RE = re.compile(r"section\s*(\d{1,2})\s*[:.\-]?\s*", re.I)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_sha: str
    source: str
    chemical_key: str
    chemical_name: str
    section: int
    section_title: str
    page_start: int
    page_end: int
    text: str

    @property
    def header(self) -> str:
        if self.section == 0:
            return f"{self.chemical_name} | {self.section_title}"
        return f"{self.chemical_name} | Section {self.section}: {self.section_title}"

    @property
    def search_text(self) -> str:
        """Text used for embeddings and keyword search (header + body)."""
        return f"{self.header}\n{self.text}"

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}–{self.page_end}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Chunk:
        return cls(**d)  # type: ignore[arg-type]


def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def remove_boilerplate(pages: list[str], head: int = 8, tail: int = 3) -> list[str]:
    """Drop running headers/footers that repeat on most pages.

    Only lines near the top or bottom of a page are candidates, so genuine
    content that happens to repeat (e.g. "Not available.") is preserved.
    """
    n = len(pages)
    if n < 3:
        return pages
    threshold = max(3, math.ceil(0.5 * n))
    counts: Counter[str] = Counter()
    per_page: list[list[str]] = []
    for page in pages:
        lines = page.splitlines()
        per_page.append(lines)
        idx = [i for i, ln in enumerate(lines) if ln.strip()]
        edge = set(idx[:head]) | set(idx[-tail:])
        counts.update({_norm_line(lines[i]) for i in edge})
    repeated = {k for k, c in counts.items() if c >= threshold and k}

    cleaned = []
    for lines in per_page:
        idx = [i for i, ln in enumerate(lines) if ln.strip()]
        edge = set(idx[:head]) | set(idx[-tail:])
        keep = [ln for i, ln in enumerate(lines) if not (i in edge and _norm_line(ln) in repeated)]
        cleaned.append("\n".join(keep))
    return cleaned


def _valid_heading(num: int, following: str) -> bool:
    if num not in _SECTION_KEYWORDS:
        return False
    head = following.lower().lstrip(" :.-")
    return any(head.startswith(k) for k in _SECTION_KEYWORDS[num])


def segment_sections(pages: list[str]) -> list[tuple[int, list[tuple[str, int]]]]:
    """Split page text into ``(section_number, [(line, page_no), ...])`` segments.

    Headings may appear mid-line (some PDF layouts glue them to the previous
    text). Section numbers must not decrease, so repeated running headers
    simply continue the current section and backward references are ignored.
    """
    segments: list[tuple[int, list[tuple[str, int]]]] = [(0, [])]
    current = 0
    for page_no, page in enumerate(pages, start=1):
        for line in page.splitlines():
            rest = line
            search_from = 0
            while True:
                found = None
                for m in _HEADING_RE.finditer(rest, search_from):
                    num = int(m.group(1))
                    if num > current and _valid_heading(num, rest[m.end() :]):
                        found = m
                        break
                if found is None:
                    if rest.strip():
                        segments[-1][1].append((rest, page_no))
                    break
                before = rest[: found.start()]
                if before.strip():
                    segments[-1][1].append((before, page_no))
                current = int(found.group(1))
                segments.append((current, []))
                rest = rest[found.start() :]
                search_from = found.end() - found.start()
    return [(num, lines) for num, lines in segments if any(t.strip() for t, _ in lines)]


def _split_long(text: str, limit: int) -> list[str]:
    """Split an over-long line at sentence, then word, boundaries."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    buf = ""
    for sent in re.split(r"(?<=[.;:])\s+", text):
        while len(sent) > limit:
            cut = sent.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append((buf + " " + sent[:cut]).strip() if buf else sent[:cut])
            buf = ""
            sent = sent[cut:].lstrip()
        if len(buf) + len(sent) + 1 > limit and buf:
            pieces.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def chunk_document(
    pages: list[str],
    identity: ChemicalIdentity,
    source: str,
    doc_sha: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Produce section-aligned chunks for one SDS."""
    cleaned = remove_boilerplate(pages)
    segments = segment_sections(cleaned)
    chunks: list[Chunk] = []

    for section, lines in segments:
        units: list[tuple[str, int]] = []
        for text, page in lines:
            text = re.sub(r"[ \t]+", " ", text).strip()
            if text:
                units.extend((piece, page) for piece in _split_long(text, chunk_size))

        for group in _pack_units(units, chunk_size, chunk_overlap):
            body = "\n".join(t for t, _ in group).strip()
            if not body:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_sha[:12]}-{len(chunks):04d}",
                    doc_sha=doc_sha,
                    source=source,
                    chemical_key=identity.key,
                    chemical_name=identity.name,
                    section=section,
                    section_title=SECTION_TITLES.get(section, f"Section {section}"),
                    page_start=min(p for _, p in group),
                    page_end=max(p for _, p in group),
                    text=body,
                )
            )
    return chunks


def _pack_units(
    units: list[tuple[str, int]], chunk_size: int, chunk_overlap: int
) -> list[list[tuple[str, int]]]:
    """Greedily pack lines into groups of at most ``chunk_size`` characters.

    Each new group starts with trailing lines of the previous one (up to
    ``chunk_overlap`` characters) so facts spanning a boundary stay intact.
    """
    groups: list[list[tuple[str, int]]] = []
    buf: list[tuple[str, int]] = []
    buf_len = 0
    for text, page in units:
        if buf and buf_len + len(text) + 1 > chunk_size:
            groups.append(buf)
            carry: list[tuple[str, int]] = []
            carry_len = 0
            for t, p in reversed(buf):
                if carry_len + len(t) + 1 > chunk_overlap:
                    break
                carry.insert(0, (t, p))
                carry_len += len(t) + 1
            buf, buf_len = carry, carry_len
        buf.append((text, page))
        buf_len += len(text) + 1
    if buf:
        groups.append(buf)
    return groups
