"""Reproducible RAG evaluation on eval/questions.json.

Retrieval mode (no LLM needed) compares retrieval configurations:

    python scripts/evaluate.py retrieval
    python scripts/evaluate.py retrieval --models sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5 \
        --chunk-sizes 800,1000,1400 --rerank

Answer mode runs the full pipeline against the configured Ollama model:

    python scripts/evaluate.py answers [--limit 10]

Results are written to eval/results/. Answer metrics are automated proxies
(fact-string matching, refusal detection, citation validity); they do not
replace human review of answer quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import Settings, configure_logging, load_settings  # noqa: E402
from rag.embeddings import Embedder  # noqa: E402
from rag.evaluation import (  # noqa: E402
    EvalQuestion,
    QuestionResult,
    RetrievedRef,
    load_questions,
    score_retrieval,
    summarise,
)
from rag.indexing import KnowledgeBase, load_knowledge_base, sync_knowledge_base  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402

KS = (1, 3, 5)
QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "eval" / "results"
log = logging.getLogger("evaluate")


def chemical_sources(kb: KnowledgeBase) -> dict[str, str]:
    return {d.chemical.key: d.filename for d in kb.manifest.documents}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_eval_index(base: Settings, model: str, chunk_size: int) -> tuple[Settings, Embedder, KnowledgeBase]:
    overlap = min(base.chunk_overlap, chunk_size // 4)
    s = base.with_overrides(
        embedding_model=model,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        index_dir=ROOT / "storage" / "eval" / f"{slug(model)}-{chunk_size}",
    )
    embedder = Embedder(model, s.embedding_device)
    t0 = time.perf_counter()
    report = sync_knowledge_base(s, embedder)
    log.info("index %s/%s: %s (%.1fs)", model, chunk_size, report.summary(), time.perf_counter() - t0)
    return s, embedder, load_knowledge_base(s)


def run_retrieval(
    questions: list[EvalQuestion], retriever: Retriever, kb: KnowledgeBase
) -> tuple[list[QuestionResult], float]:
    chem = chemical_sources(kb)
    results = []
    t0 = time.perf_counter()
    for q in questions:
        res = retriever.retrieve(q.question, top_k=max(KS))
        refs = [RetrievedRef(r.chunk.source, r.chunk.page_start, r.chunk.page_end) for r in res.chunks]
        results.append(score_retrieval(q, refs, KS, chem))
    return results, (time.perf_counter() - t0) / max(1, len(questions))


def cmd_retrieval(args: argparse.Namespace) -> None:
    base = load_settings()
    questions = load_questions(QUESTIONS)
    models = args.models.split(",") if args.models else [base.embedding_model]
    sizes = [int(x) for x in args.chunk_sizes.split(",")] if args.chunk_sizes else [base.chunk_size]
    reranker = None
    if args.rerank:
        from rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(base.reranker_model)

    rows = []
    for model in models:
        for size in sizes:
            s, embedder, kb = build_eval_index(base, model, size)
            variants: list[tuple[str, dict[str, bool], bool, Any]] = [
                ("dense, no chemical filter", dict(hybrid_search=False), False, None),
                ("dense + chemical filter", dict(hybrid_search=False), True, None),
                ("hybrid, no chemical filter", dict(hybrid_search=True), False, None),
                ("hybrid + chemical filter", dict(hybrid_search=True), True, None),
            ]
            if reranker is not None:
                variants.append(("hybrid + filter + rerank", dict(hybrid_search=True), True, reranker))
            for name, overrides, filt, rr in variants:
                retriever = Retriever(kb, embedder, s.with_overrides(**overrides), rr, chemical_filtering=filt)
                results, latency = run_retrieval(questions, retriever, kb)
                summ = summarise(results, KS)
                summ["retrieval_latency_ms"] = round(latency * 1000, 1)
                rows.append({"model": model, "chunk_size": size, "chunks": len(kb.chunks),
                             "variant": name, **summ,
                             "misses@5": [r.id for r in results if r.hit_at and not r.hit_at[5]]})
                print(f"{model:45s} {size:5d} {name:28s} hit@1={summ.get('hit@1')} hit@3={summ.get('hit@3')} "
                      f"hit@5={summ.get('hit@5')} mrr={summ.get('mrr')} contam={summ.get('contamination@k')} "
                      f"{summ['retrieval_latency_ms']}ms", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / (args.output or "retrieval_comparison.json")
    out.write_text(json.dumps({"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                               "questions": len(questions), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")


def cmd_answers(args: argparse.Namespace) -> None:
    from rag.pipeline import RagPipeline

    settings = load_settings()
    questions = load_questions(QUESTIONS)
    if args.limit:
        questions = questions[: args.limit]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q.id in wanted]
    pipeline = RagPipeline.from_settings(settings)
    chem = chemical_sources(pipeline.kb)

    from rag.evaluation import facts_match, is_refusal

    results: list[QuestionResult] = []
    for q in questions:
        t0 = time.perf_counter()
        prepared = pipeline.prepare(q.question, history=[])
        refs = [RetrievedRef(s.source, s.page_start, s.page_end) for s in prepared.sources]
        r = score_retrieval(q, refs, KS, chem)
        try:
            answer = pipeline.complete(prepared)
            r.latency_s = time.perf_counter() - t0
            r.answer = answer.text
            r.refused = answer.refused or is_refusal(answer.text)
            if q.answerable:
                r.facts_ok = facts_match(answer.text, q.facts) if q.facts else None
            else:
                r.refusal_correct = r.refused
            allowed = {chem[c] for c in q.chemicals if c in chem}
            r.citations_total = len(answer.cited_labels) + len(answer.invalid_labels)
            r.citations_valid = len(answer.cited_labels)
            r.citations_on_target = sum(
                1 for s in answer.cited_sources if not allowed or Path(s.source).name in allowed
            )
            r.uncited = answer.uncited
        except Exception as exc:
            r.error = f"{exc.__class__.__name__}: {exc}"
            log.error("%s failed: %s", q.id, r.error)
        results.append(r)
        print(f"{q.id:24s} {r.latency_s or 0:6.1f}s facts={r.facts_ok} refused={r.refused} "
              f"cites={r.citations_valid}/{r.citations_total}", flush=True)

    summary = summarise(results, KS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / (args.output or "answers_current.json")
    out.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm": settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "settings": {k: str(v) for k, v in asdict(settings).items() if "dir" not in k},
        "summary": summary,
        "per_question": [r.__dict__ for r in results],
    }, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(ROOT)}")


def cmd_rescore(args: argparse.Namespace) -> None:
    """Recompute fact/refusal metrics of a saved answers file with the current scoring code."""
    from rag.evaluation import facts_match, is_refusal

    path = RESULTS / args.file
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = {q.id: q for q in load_questions(QUESTIONS)}
    results = []
    for row in data["per_question"]:
        r = QuestionResult(**{k: v for k, v in row.items() if k in QuestionResult.__dataclass_fields__})
        r.hit_at = {int(k): v for k, v in r.hit_at.items()}
        q = qs[r.id]
        if r.answer is not None:
            r.refused = is_refusal(r.answer) and not (r.citations_valid or 0)
            r.facts_ok = facts_match(r.answer, q.facts) if q.answerable and q.facts else None
            r.refusal_correct = None if q.answerable else is_refusal(r.answer)
            r.uncited = not (r.citations_total or 0) and not r.refused
        results.append(r)
    data["summary"] = summarise(results, KS)
    data["per_question"] = [r.__dict__ for r in results]
    data["rescored_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(json.dumps(data["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval", help="compare retrieval configurations (no LLM)")
    r.add_argument("--models", help="comma-separated embedding models")
    r.add_argument("--chunk-sizes", help="comma-separated chunk sizes")
    r.add_argument("--rerank", action="store_true", help="also evaluate the cross-encoder reranker")
    r.add_argument("--output", help="output file name in eval/results/")
    a = sub.add_parser("answers", help="full pipeline with the configured Ollama model")
    a.add_argument("--limit", type=int, help="only the first N questions")
    a.add_argument("--only", help="comma-separated question ids")
    a.add_argument("--output", help="output file name in eval/results/")
    rs = sub.add_parser("rescore", help="re-score a saved answers file with the current metrics")
    rs.add_argument("file", help="file name in eval/results/")
    args = parser.parse_args()
    configure_logging("INFO")
    logging.getLogger("evaluate").setLevel(logging.INFO)
    {"retrieval": cmd_retrieval, "answers": cmd_answers, "rescore": cmd_rescore}[args.cmd](args)


if __name__ == "__main__":
    main()
