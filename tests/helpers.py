"""Test doubles and a dependency-free PDF writer."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

import numpy as np

from rag.llm import Message


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]], title: str = "") -> bytes:
    """Build a minimal text PDF (Helvetica, one line per entry) that pypdf can read."""
    objects: list[bytes] = []

    def add(obj: str | bytes) -> int:
        objects.append(obj.encode("latin-1") if isinstance(obj, str) else obj)
        return len(objects)

    catalog = add("")  # placeholder, filled below
    pages_obj = add("")
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    for lines in pages:
        ops = ["BT", "/F1 9 Tf", "40 760 Td", "11 TL"]
        for i, line in enumerate(lines):
            ops.append(f"({_pdf_escape(line)}) Tj" if i == 0 else f"T* ({_pdf_escape(line)}) Tj")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(add(
            f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"
        ))
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects[pages_obj - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    info = add(f"<< /Title ({_pdf_escape(title)}) >>") if title else None

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    trailer = f"<< /Size {len(objects) + 1} /Root {catalog} 0 R"
    if info:
        trailer += f" /Info {info} 0 R"
    out += f"trailer\n{trailer} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def sds_pages(name: str, cas: str, flash: str, extra: dict[int, list[str]] | None = None,
              synonym: str = "") -> list[list[str]]:
    """A synthetic 4-page SDS with all 16 section headings and a running header."""
    extra = extra or {}
    hdr = [f"Safety data sheet - {name}", "Version 1.0"]
    sec = {n: extra.get(n, []) for n in range(1, 17)}
    p1 = [*hdr, "SECTION 1: Identification of the substance/mixture", f"Product name : {name}", f"CAS-No. : {cas}", *([f"Synonyms : {synonym}"] if synonym else []), *sec[1], "SECTION 2: Hazards identification", "Signal word Danger", "H225 Highly flammable liquid and vapour.", *sec[2], "SECTION 3: Composition/information on ingredients", f"Name of substance {name}", *sec[3], "SECTION 4: First aid measures", "Following eye contact rinse with water for at least 10 minutes.", *sec[4], "Page 1 / 4"]
    p2 = [*hdr, "SECTION 5: Firefighting measures", "Unsuitable extinguishing media: water jet.", "SECTION 6: Accidental release measures", "Absorb with liquid-binding material.", "SECTION 7: Handling and storage", "Keep container tightly closed. Incompatible materials: see section 10. Keep away from oxidisers.", *sec[7], "SECTION 8: Exposure controls/personal protection", "Wear butyl rubber gloves, breakthrough time >480 minutes.", *sec[8], "Page 2 / 4"]
    p3 = [*hdr, "SECTION 9: Physical and chemical properties", f"Flash point {flash}", *sec[9], "SECTION 10: Stability and reactivity", "Stable under normal conditions. See section 2: Hazards identification for details.", "SECTION 11: Toxicological information", *(sec[11] or ["No data available."]), "SECTION 12: Ecological information", "No data available.", "Page 3 / 4"]
    p4 = [*hdr, "SECTION 13: Disposal considerations", "Dispose of as hazardous waste.", "SECTION 14: Transport information", *(sec[14] or ["UN number: 0000"]), "SECTION 15: Regulatory information", "Listed.", "SECTION 16: Other information", *(sec[16] or ["None."]), "Page 4 / 4"]
    return [p1, p2, p3, p4]


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download)."""

    def __init__(self, model_name: str = "test/fake-embedder", dimension: int = 256) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.revision = "test"
        self.query_prefix = ""
        self.passage_prefix = ""
        self.documents_embedded = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dimension] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts, batch_size: int = 32, progress=None) -> np.ndarray:
        self.documents_embedded += len(texts)
        if progress:
            progress(len(texts), len(texts))
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)[None, :]


class FakeLLM:
    """Scripted chat model. ``reply`` may be a string or a callable(messages)->str."""

    def __init__(self, reply="[S1] answer.", error: Exception | None = None,
                 error_after: int | None = None) -> None:
        self.reply = reply
        self.error = error
        self.error_after = error_after
        self.calls: list[list[Message]] = []

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        self.calls.append(messages)
        text = self.reply(messages) if callable(self.reply) else self.reply
        words = re.findall(r"\S+\s*", text)
        if self.error is not None and self.error_after is None:
            raise self.error
        for i, w in enumerate(words):
            if self.error is not None and self.error_after is not None and i >= self.error_after:
                raise self.error
            yield w
