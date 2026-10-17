"""Evaluation metrics shared by the evaluation runner and baseline scripts.

Everything here is pure Python so it can be unit-tested without models.
Retrieval metrics are computed against page-level ground truth; answer
metrics are automated *proxies* (fact-string matching, refusal detection,
citation validity) and are not a substitute for human review.
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.prompts import is_refusal

__all__ = ["is_refusal"]


@dataclass(frozen=True)
class Expected:
    source: str
    pages: tuple[int, ...]


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    question: str
    chemicals: tuple[str, ...]
    answerable: bool
    expected: tuple[Expected, ...]
    facts: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RetrievedRef:
    """Minimal view of a retrieved chunk used for scoring."""

    source: str
    page_start: int
    page_end: int

    def matches(self, exp: Expected) -> bool:
        if Path(self.source).name != Path(exp.source).name:
            return False
        return any(self.page_start <= p <= self.page_end for p in exp.pages)


@dataclass
class QuestionResult:
    id: str
    category: str
    hit_at: dict[int, bool] = field(default_factory=dict)
    reciprocal_rank: float = 0.0
    contamination: float | None = None
    answer: str | None = None
    facts_ok: bool | None = None
    refused: bool | None = None
    refusal_correct: bool | None = None
    citations_total: int | None = None
    citations_valid: int | None = None
    citations_on_target: int | None = None
    uncited: bool | None = None
    latency_s: float | None = None
    error: str | None = None


def load_questions(path: Path) -> list[EvalQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for q in data["questions"]:
        out.append(
            EvalQuestion(
                id=q["id"],
                category=q["category"],
                question=q["question"],
                chemicals=tuple(q.get("chemicals", [])),
                answerable=bool(q["answerable"]),
                expected=tuple(Expected(e["source"], tuple(e["pages"])) for e in q["expected"]),
                facts=tuple(tuple(g) for g in q.get("facts", [])),
            )
        )
    return out


def normalise_text(text: str) -> str:
    """Lowercase, NFKC-normalise, unify minus signs and decimal commas."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return re.sub(r"\s+", " ", text)


def facts_match(answer: str, facts: tuple[tuple[str, ...], ...]) -> bool:
    norm = normalise_text(answer)
    return all(any(normalise_text(alt) in norm for alt in group) for group in facts)


def first_relevant_rank(refs: list[RetrievedRef], exp: Expected) -> int | None:
    for i, ref in enumerate(refs, start=1):
        if ref.matches(exp):
            return i
    return None


def score_retrieval(
    q: EvalQuestion,
    refs: list[RetrievedRef],
    ks: tuple[int, ...],
    chemical_sources: dict[str, str],
) -> QuestionResult:
    """Score one question's ranked retrieval results.

    For multi-chemical questions every expected source must be found for a
    hit, and the reciprocal rank is averaged across expected sources.
    """
    res = QuestionResult(id=q.id, category=q.category)
    if q.expected:
        ranks = [first_relevant_rank(refs, e) for e in q.expected]
        for k in ks:
            res.hit_at[k] = all(r is not None and r <= k for r in ranks)
        res.reciprocal_rank = statistics.mean((1.0 / r) if r else 0.0 for r in ranks)
    allowed = {chemical_sources[c] for c in q.chemicals if c in chemical_sources}
    if allowed and refs:
        top = refs[: max(ks)]
        off = sum(1 for r in top if Path(r.source).name not in allowed)
        res.contamination = off / len(top)
    return res


def summarise(results: list[QuestionResult], ks: tuple[int, ...]) -> dict[str, Any]:
    """Aggregate per-question results into headline metrics."""
    with_expected = [r for r in results if r.hit_at]
    summary: dict[str, Any] = {"questions": len(results)}
    for k in ks:
        if with_expected:
            summary[f"hit@{k}"] = round(statistics.mean(r.hit_at[k] for r in with_expected), 3)
    if with_expected:
        summary["mrr"] = round(statistics.mean(r.reciprocal_rank for r in with_expected), 3)
    contam = [r.contamination for r in results if r.contamination is not None]
    if contam:
        summary["contamination@k"] = round(statistics.mean(contam), 3)
        summary["questions_with_any_contamination"] = sum(1 for c in contam if c > 0)

    answered = [r for r in results if r.answer is not None]
    if answered:
        facts = [r.facts_ok for r in answered if r.facts_ok is not None]
        if facts:
            summary["answer_fact_accuracy"] = round(statistics.mean(facts), 3)
        ref = [r.refusal_correct for r in answered if r.refusal_correct is not None]
        if ref:
            summary["unanswerable_handled"] = round(statistics.mean(ref), 3)
        tot = sum(r.citations_total or 0 for r in answered)
        if tot:
            summary["citation_validity"] = round(sum(r.citations_valid or 0 for r in answered) / tot, 3)
            summary["citation_on_target"] = round(
                sum(r.citations_on_target or 0 for r in answered) / tot, 3
            )
        summary["uncited_answers"] = sum(1 for r in answered if r.uncited)
        lat = [r.latency_s for r in answered if r.latency_s is not None]
        if lat:
            summary["latency_median_s"] = round(statistics.median(lat), 2)
            summary["latency_p90_s"] = round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2)
    summary["failures"] = sum(1 for r in results if r.error)
    return summary
