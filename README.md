# Chemical Safety RAG Assistant

Ask plain-English questions about chemical hazards, protective equipment, first aid, storage and
physical properties, and get answers drawn **only** from the Safety Data Sheets (SDS) in your
knowledge base, with every statement linked to the document and page it came from.

Runs fully locally: a Streamlit UI, a local LLM served by [Ollama](https://ollama.com) (Mistral 7B by
default), and a FAISS + BM25 hybrid index. No API keys, no data leaves the machine.

> **Safety notice.** Answers are generated from the SDS documents that have been indexed and can be
> incomplete or out of date. Always consult the supplier's current official SDS and follow your site
> procedures. In an emergency, call your local emergency number or poison control centre.

---

## Contents

- [What it does](#what-it-does)
- [Quick start (local)](#quick-start-local)
- [Quick start (Docker)](#quick-start-docker)
- [Using the app](#using-the-app)
- [How it works](#how-it-works)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Managing the knowledge base](#managing-the-knowledge-base)
- [Testing](#testing)
- [Deployment notes](#deployment-notes)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Known limitations](#known-limitations)
- [Changelog](#changelog)

---

## What it does

- **Grounded, cited answers.** Every retrieved excerpt is labelled (`[S1]`, `[S2]`, …) with its
  chemical, file, page and SDS section. The model cites labels; the app validates each citation
  against what was actually retrieved and renders them as numbered references `[1]`, `[2]` with an
  expandable panel showing the exact passage. Invented references are removed and flagged.
- **No cross-chemical mix-ups.** Each SDS is tagged with its chemical identity (name, CAS number,
  synonyms). Questions are matched to chemicals by name, alias (e.g. *methyl alcohol*, *2-propanone*)
  or CAS number, and retrieval is restricted to those SDSs. Multi-chemical questions retrieve evidence
  for each chemical separately.
- **Honest about gaps.** Questions about a chemical with no SDS in the knowledge base (e.g. toluene)
  get a clear "no SDS" answer without calling the model; missing information is stated, not guessed.
- **SDS-aware indexing.** Documents are split along the 16 standard SDS sections, running headers and
  footers are removed, and every chunk keeps its exact page range.
- **Hybrid retrieval.** Dense vectors (FAISS) plus BM25 keyword search fused with Reciprocal Rank
  Fusion, so exact identifiers such as CAS numbers, UN numbers and H-statements are found reliably.
- **Live knowledge base.** Upload SDS PDFs in the app; they are validated, de-duplicated and indexed
  incrementally. The index is versioned and swapped atomically, so a failed update never corrupts it.
- **Streaming answers**, a lightweight Ollama health check with retry, conversation export to
  Markdown, light and dark themes.

## Quick start (local)

**Prerequisites:** Python 3.11+ (3.12 tested), [Ollama](https://ollama.com/download), about 8 GB of
free RAM for Mistral 7B on CPU (a GPU is optional and makes answers much faster).

```bash
# 1. Get the model (once, ~4.4 GB)
ollama pull mistral

# 2. Create a virtual environment and install dependencies
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
# Linux without a GPU: install the much smaller CPU build of torch first
#   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0

# 3. Build the index from the PDFs in data/ (~20 s; downloads the 90 MB embedding model once)
python ingest.py

# 4. Start the app
streamlit run app.py
```

Open <http://localhost:8501>. If you skip step 3, the app builds the index on first start.

Optional: `cp .env.example .env` and edit values (see [Configuration](#configuration)).

## Quick start (Docker)

The compose file runs two containers: the app and an Ollama server. The model is **not** bundled in
any image; pull it once into the persistent `ollama-models` volume.

```bash
docker compose up -d --build
docker compose exec ollama ollama pull mistral    # once
# open http://localhost:8501
```

- Documents and uploads persist in the `sds-documents` volume (seeded with the bundled PDFs on first
  start); the index persists in `sds-index`. The container brings the index up to date on start.
- The embedding model is baked into the app image at build time, so the app needs no internet access
  at runtime.
- NVIDIA GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` (requires the
  NVIDIA Container Toolkit).
- Stop with `docker compose down` (volumes are kept; `docker compose down -v` deletes them).

## Using the app

| Feature | Where |
|---|---|
| Ask a question | Chat box at the bottom, or click an example question |
| See evidence | **Sources** panel under each answer: numbered passages with file, page and SDS section |
| Restrict to certain chemicals | Sidebar → *Limit answers to* (by default chemicals are detected from your question) |
| Follow-up questions | "What about its UN number?" keeps the chemical from the previous question |
| Add documents | Sidebar → *Add SDS documents* → choose PDFs → *Add to knowledge base* |
| Model offline? | Sidebar shows the Ollama status with a *Check again* button; no reload needed |
| Clear / export chat | Sidebar → *Conversation*. Clearing never touches the knowledge base |
| Theme | Follows your system setting; switch in the ⋮ menu → Settings |

Uploads are checked for type, size (`MAX_UPLOAD_MB`), page count, encryption and extractable text.
Scanned image-only PDFs are rejected (OCR is not supported). Re-uploading a file with the same name
replaces the earlier upload; uploading content that is already indexed is reported as a duplicate.

## How it works

```
                 ┌──────────── ingestion (python ingest.py / in-app upload) ────────────┐
 SDS PDFs ──► pypdf text ──► chemical identity ──► boilerplate removal ──► 16-section split
 (data/)      per page       (name, CAS, synonyms)   (running headers)      + sub-chunks with page ranges
                                                                                   │
                         embeddings (all-MiniLM-L6-v2) ──► FAISS IndexFlatIP  ◄────┘
                         manifest.json (model, chunking, file hashes) + chunks.json
                 └───────────────────────────────────────────────────────────────────────┘

 question ──► detect chemicals ──► scoped search per chemical ──► dense + BM25 ──► RRF fusion
                (names, aliases,      (or history / user filter)                    top-k excerpts
                 CAS numbers)                                                            │
              prompt: rules + <sources>[S1] Chemical | File | page | Section … </sources> ◄┘
                                   │
                    Ollama (streamed) ──► citation validation ──► answer with [1][2] + sources panel
```

Key design decisions:

- **No LangChain chains.** The pipeline is plain, typed Python (`rag/`) on top of
  `pypdf`, `sentence-transformers`, `faiss-cpu` and Ollama's REST API via `httpx`. The previous
  version depended on fast-moving LangChain chain APIs (it needed a `langchain`/`langchain_classic`
  import fallback) and loaded its index with pickle. Custom chemical scoping, hybrid search, label
  citations and streaming-with-sources are simpler and fully testable this way.
- **No pickle.** The index is `index.faiss` + `chunks.json` + `manifest.json`; loading it cannot
  execute code.
- **Citations by label, not by model-written file names.** The model only ever sees and writes
  `[S#]` labels; file names and pages come from the retriever's metadata.
- **Documents are untrusted.** Excerpt text is fenced inside `<sources>`, label-like strings and tag
  look-alikes in PDF text are neutralised, the system prompt tells the model to ignore instructions
  in excerpts, and rendered answers have images, raw HTML and remote links neutralised.

## Evaluation

`eval/questions.json` holds 51 questions written against the four bundled SDSs, with ground-truth
pages verified by reading the extracted text of each PDF: identifiers, numeric properties with units,
hazards, first aid, storage, exposure controls, aliases, CAS look-ups, multi-chemical comparisons,
multi-section questions, "the SDS says no data" cases and unanswerable questions.

```bash
python scripts/evaluate.py retrieval                      # retrieval metrics, no LLM needed
python scripts/evaluate.py retrieval --models sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5 \
       --chunk-sizes 800,1000,1400 --rerank                # comparison grid
python scripts/evaluate.py answers                        # full pipeline against Ollama (~1 h on CPU)
python scripts/evaluate.py answers --only multi-un,cross-section-ace,unans-price,unans-eth-vp50,unans-toluene
```

The answer run needs roughly 6 GB of free RAM for Mistral on top of the app; close other heavy
applications first.

Results are stored in `eval/results/`.

### Baseline vs current (measured on this machine: 12-core CPU, no GPU, Mistral 7B Q4)

Retrieval (all 51 questions, no LLM involved):

| Metric | Original v2 pipeline | Current |
|---|---|---|
| Retrieval hit@1 | 0.327 | 0.837 |
| Retrieval hit@3 | 0.551 | 0.980 |
| Retrieval hit@5 | 0.653 | 1.000 |
| Mean reciprocal rank | 0.469 | 0.926 |
| Retrieved chunks from the wrong chemical (top 5) | 30% | 0% |
| Questions with any wrong-chemical chunk | 31 / 51 | 0 / 51 |

Answers with Mistral 7B. The full v3 answer run was stopped by the host after **46 of 51** questions
(the machine ran out of memory); `multi-un`, `cross-section-ace`, `unans-price`, `unans-eth-vp50` and
`unans-toluene` were not in it. Both columns below cover the same 46 questions
(`eval/results/answers_v3_partial.json`, `eval/results/legacy_baseline_answers.json`):

| Metric (same 46 questions) | Original v2 pipeline | Current |
|---|---|---|
| Answer fact accuracy (automated) | 0.565 | **0.913** |
| Answers without any citation | 16 / 46 | **0 / 46** |
| Citations pointing at a source that was actually retrieved | 0.781 (parseable subset) | **1.000** (71 / 71) |
| Median answer latency (CPU) | 42.7 s | 41.0 s* |

\* The v3 latency was measured while other jobs shared the CPU, so it is indicative only.
Refusals: the toluene question (no SDS indexed) is declined without calling the model, verified by
`answers_smoke.json` and the live test; the other two unanswerable questions were not re-run on v3
(the v2 pipeline declined all three). The four v3 misses were `ace-oel`, `ben-ld50`, `met-glove` and
`multi-ld50`; v2 missed 20 of the same questions.

How to read this:

- *Retrieval* metrics check whether a chunk from the right document **and page** is in the top *k*.
  For multi-chemical questions every chemical's page must be found.
- *Answer fact accuracy* checks that the expected values (e.g. `-17`, `butyl`, `H225`) appear in the
  answer. It is an automated proxy, not a human judgement of answer quality.
- The original pipeline asked the model to write file names and pages itself; only some answers used
  a parseable format, so its citation numbers cover a subset of answers. In the current pipeline every
  citation is a validated label.
- The evaluation set was written for and used during development (for example, tokeniser fixes were
  made after inspecting retrieval misses), so these are development-set numbers on four documents,
  not a held-out benchmark.

### Retrieval design choices (from `eval/results/retrieval_comparison.json`)

| Variant (MiniLM, chunk size 1000) | hit@1 | hit@5 | MRR | Wrong-chemical chunks |
|---|---|---|---|---|
| Dense only | 0.776 | 0.939 | 0.845 | 8.4% |
| Dense + chemical filter | 0.796 | 0.959 | 0.874 | 0% |
| Hybrid (dense + BM25), no filter | 0.857 | 0.980 | 0.926 | 22.8% |
| **Hybrid + chemical filter (default)** | 0.837 | **1.000** | 0.926 | **0%** |
| Hybrid + filter + cross-encoder reranker | 0.898 | 1.000 | 0.961 | 0% |

- **Embedding model:** BGE-small-en-v1.5 (MRR 0.937) and E5-small-v2 (MRR 0.940) were within one or
  two questions of all-MiniLM-L6-v2 (MRR 0.926) with the same hit@5. MiniLM was kept: it is the
  smallest and fastest, and the difference is within noise on 49 scored questions.
- **Reranker:** improves ordering (hit@1 0.84 → 0.90) but adds ~1.4 s per query on CPU and a 90 MB
  model, while hit@5 is already 1.0 and the model sees all six excerpts. It is available
  (`RERANKER_ENABLED=true`) but off by default.
- **Chunk size:** 800, 1000 and 1400 characters performed within one question of each other; 1000
  was kept.

## Configuration

All settings are environment variables (or a `.env` file; see `.env.example`). Invalid values stop
the app with a message naming the variable.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `mistral` | Model name as shown by `ollama list` |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_NUM_CTX` | `8192` | Context window passed to Ollama |
| `LLM_CONNECT_TIMEOUT_S` / `LLM_READ_TIMEOUT_S` | `5` / `180` | Request timeouts |
| `HEALTH_TIMEOUT_S` | `3` | Health-check timeout |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Changing it requires `python ingest.py --rebuild` |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` if available |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters; changing them requires a rebuild |
| `TOP_K` | `6` | Excerpts sent to the model |
| `CANDIDATE_K` | `30` | Candidates per retriever before fusion |
| `HYBRID_SEARCH` | `true` | Dense + BM25 fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RERANKER_ENABLED` / `RERANKER_MODEL` | `false` / `cross-encoder/ms-marco-MiniLM-L-6-v2` | Optional reranking |
| `MAX_HISTORY_TURNS` | `4` | Conversation turns replayed to the model |
| `DOCS_DIR` | `data` | PDFs (recursively). Uploads go to `DOCS_DIR/uploads` |
| `INDEX_DIR` | `storage/index` | Generated index |
| `MAX_UPLOAD_MB` / `MAX_PDF_PAGES` / `MAX_QUESTION_CHARS` | `20` / `200` / `2000` | Resource limits |
| `LOG_LEVEL` | `INFO` | Logs record timings and counts, not question text (use `DEBUG` for more) |

## Managing the knowledge base

```bash
python ingest.py            # incremental: embeds only new or changed PDFs, drops deleted ones
python ingest.py --rebuild  # re-embed everything (after changing EMBEDDING_MODEL or CHUNK_*)
python ingest.py --check    # report status; exit code 1 if the index is missing, stale or incompatible
```

- Add PDFs by copying them into `data/` (or uploading in the app), then run `python ingest.py` or use
  the *Update index* button the app shows when it detects changes.
- Documents are identified by SHA-256: edited files are re-embedded, renamed files keep their vectors,
  byte-identical copies are indexed once (the copy is reported, not treated as a pending change).
- If one file in a multi-file upload cannot be indexed, only that file is rolled back; the others
  stay indexed.
- Each build writes `storage/index/versions/<version>/` and then atomically updates
  `storage/index/CURRENT`. The previous version is kept; failed builds leave the current index in use.
- `manifest.json` records the embedding model and revision, chunking settings, pipeline version,
  library versions and each document's hash, chemical identity and page count. The app refuses an
  index built with a different embedding model and warns when chunking settings differ.
- **Migrating from v2:** the old `vectorstore/` (LangChain pickle format) is no longer read. The new
  index is built from `data/` automatically. Once you're happy with v3 you can remove the old files
  with `git rm -r vectorstore`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -m "not live"        # unit, integration, UI (Streamlit AppTest) and real-PDF regression tests
pytest -m live              # end-to-end against your running Ollama (skipped if unreachable)
ruff check . && mypy        # lint and type check
```

- Unit and integration tests use synthetic PDFs generated in-process and deterministic fake
  embedding/LLM doubles, so they run in seconds without models.
- `tests/test_real_corpus.py` checks identity extraction, section detection and page numbers on the
  four bundled PDFs, plus a retrieval-quality regression with the real embedding model (marked
  `slow`, skipped if the model isn't cached).
- `tests/test_app.py` drives the real `app.py` with Streamlit's `AppTest`.
- `scripts/ui_smoke.py` checks the running app in headless Chrome (render, dark mode, phone width,
  keyboard focus, optionally a real streamed answer) and saves screenshots:
  `python scripts/ui_smoke.py http://localhost:8501 ui-shots --ask "What is the flash point of acetone?"`
- CI: `.github/workflows/ci.yml` runs lint, type checks and `pytest -m "not live"`.

## Deployment notes

- **Resources.** App image: ~2.9 GB on disk (CPU-only torch, embedding model baked in). App
  container: ~1.5 GB RAM (embedding model, index, Streamlit). Ollama with
  Mistral 7B Q4: ~6 GB RAM with the default 8k context. CPU-only works; expect 20–90 s per answer on
  a modern laptop CPU, a few seconds on a GPU.
- **Persistent storage is required** for the documents directory (uploads) and the index. Without it,
  uploads disappear on restart (the bundled documents and index are rebuilt automatically).
- **Hosting compatibility.** The app needs a reachable Ollama server. Platforms such as Streamlit
  Community Cloud cannot run Ollama or a 7B model, and their filesystems are ephemeral, so they are
  not suitable for the full stack. Use a VM or container host with enough RAM (or a GPU), run the
  compose stack, and put it behind a reverse proxy with authentication if it is exposed beyond a
  trusted network. The app itself has no user accounts.
- **Security.** No secrets are needed. Uploads are size/type/page-limited and parsed in memory before
  being written atomically; error details are logged server-side and not shown to users
  (`showErrorDetails = "none"`); XSRF protection is on; index writes are serialised with a lock.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar: *Ollama · offline* | Start Ollama (`ollama serve` or the desktop app), then *Check again* |
| *model not installed* | `ollama pull mistral` (or set `OLLAMA_MODEL` to a model from `ollama list`) |
| *Index was built with embedding model …* | You changed `EMBEDDING_MODEL`: run `python ingest.py --rebuild` or click *Rebuild index* |
| *Documents folder changed since indexing* | Click *Update index* or run `python ingest.py` |
| *The knowledge base is already being updated* | Another session or `ingest.py` is writing; wait and retry. A lock older than 30 min is cleared automatically |
| Upload rejected as scanned | The PDF has no text layer; OCR it first or use a text-based SDS |
| Answers are slow | Normal on CPU. Use a GPU, a smaller model, or lower `TOP_K` |
| First start is slow | The embedding model (~90 MB) is downloaded once and the index is built |

## Project structure

```
app.py                 Streamlit UI (state + rendering only)
ingest.py              CLI: build / sync / check the index
rag/
  config.py            environment-driven settings + validation
  loader.py            PDF validation and text extraction
  chemicals.py         chemical identity extraction and question matching (names, aliases, CAS)
  chunking.py          boilerplate removal, SDS section detection, page-tracked chunks
  embeddings.py        sentence-transformers wrapper
  indexing.py          manifest, incremental sync, atomic versioned index, locking, status
  bm25.py              keyword index
  retrieval.py         chemical-scoped hybrid retrieval (FAISS + BM25 + RRF, optional reranker)
  rerank.py            optional cross-encoder
  citations.py         source labels, context formatting, citation validation/rendering
  prompts.py           system prompt and canned answers
  pipeline.py          prepare → stream → finalize
  llm.py / health.py   Ollama streaming client and health check
  uploads.py           upload validation, staging and rollback
  export.py            Markdown conversation export
  evaluation.py        retrieval and answer metrics
ui/                    rendering helpers and CSS
scripts/evaluate.py    evaluation runner
scripts/ui_smoke.py    headless-Chrome smoke test of the running app
eval/                  questions.json (ground truth) and results/
tests/                 pytest suite
data/                  bundled SDS PDFs (uploads go to data/uploads/, git-ignored)
storage/               generated index (git-ignored)
.streamlit/config.toml theme (light + dark) and server settings
Dockerfile, docker-compose.yml, docker-compose.gpu.yml, docker/entrypoint.sh
```

## Known limitations

- **PDF layout.** Text comes from `pypdf`. Some layouts (e.g. the bundled benzene SDS) extract with
  words glued together and blocks out of reading order; retrieval copes, but a fact can land in the
  neighbouring section's chunk. Page numbers remain correct.
- **No OCR.** Scanned SDSs are rejected.
- **Small evaluation set.** 51 questions over 4 documents, used during development; answer accuracy
  is measured with automated string checks.
- **One chemical per SDS.** Mixtures are indexed under their product name; component-level questions
  are not mapped to individual ingredients.
- **Follow-ups** carry the chemical over from the previous question; they are not rewritten by the LLM.
- **No stop button** while an answer streams; submitting a new question interrupts it, and the cut-off
  answer is replaced by an "interrupted" note in the conversation.
- **Phones:** Streamlit scrolls chat pages to the bottom on load, so on a phone the page opens just
  below the title (the safety notice and examples are in view; swipe up for the title).
- Fonts load from Google Fonts when online and fall back to system fonts offline.

## Changelog

### v3.0
- Rebuilt the RAG pipeline (`rag/` package) without LangChain chains or pickle.
- Fixed fabricated citations: excerpts are labelled with real file/page/section metadata and every
  citation is validated.
- Chemical identity extraction and chemical-scoped retrieval: wrong-chemical chunks 30% → 0%.
- SDS-section chunking, boilerplate removal, page-range tracking; hybrid FAISS + BM25 retrieval.
- Versioned index with manifest, incremental content-hash sync, atomic swap and write lock.
- Streaming answers, HTTP health check with retry, upload-to-knowledge-base workflow, Markdown export,
  redesigned light/dark UI, safety notice.
- Evaluation set and runner with baseline comparison; 177 automated tests (plus 2 live tests); Docker
  and CI.
- Fixes found during final verification: byte-identical PDF copies no longer leave the index
  permanently "stale"; a failed file in a multi-file upload no longer rolls back files that were
  indexed successfully; Export/Clear now include the latest answer; an answer cut off by a new
  question is marked as interrupted; visible keyboard focus on expanders; shorter empty state on phones.

### v2.0
- Ollama health check, history windowing, source de-duplication, CLI ingest, cached LLM, restyled UI.
