"""Export a conversation to Markdown with its citations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rag.citations import Source

DISCLAIMER = (
    "Answers are generated from the Safety Data Sheets in this knowledge base and may be "
    "incomplete. Always consult the current official SDS from the supplier and follow your "
    "organisation's procedures. In an emergency, contact your local emergency number or "
    "poison control centre."
)


@dataclass
class ExportMessage:
    role: str
    content: str  # rendered markdown with [n] markers
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def conversation_to_markdown(
    messages: list[ExportMessage],
    *,
    model: str,
    index_version: str,
    generated_at: datetime | None = None,
) -> str:
    ts = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Chemical Safety Assistant: conversation export",
        "",
        f"- Exported: {ts}",
        f"- Model: {model}",
        f"- Knowledge-base version: {index_version}",
        "",
        f"> {DISCLAIMER}",
        "",
    ]
    q = 0
    for msg in messages:
        if msg.role == "user":
            q += 1
            lines += ["---", "", f"## Question {q}", "", msg.content.strip(), ""]
            continue
        lines += ["### Answer", "", msg.content.strip(), ""]
        for w in msg.warnings:
            lines += [f"> **Note:** {w}", ""]
        if msg.sources:
            lines += ["**Sources**", ""]
            for i, s in enumerate(msg.sources, start=1):
                lines.append(f"{i}. {s.filename}, {s.page_label}, {s.chemical_name}, {s.section_label}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
