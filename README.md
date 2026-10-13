<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
<img src="https://img.shields.io/badge/Ollama-Mistral_7B-000000?style=for-the-badge&logo=ollama&logoColor=white"/>
<img src="https://img.shields.io/badge/FAISS-CPU-00599C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>

<br/><br/>

# ⚗️ Chemical Safety RAG Assistant

**Ask natural-language questions about chemical hazards, PPE, first aid, and storage — and get grounded, cited answers from real Safety Data Sheets.**

*Fully offline · No API keys · Powered by Mistral 7B via Ollama*

<br/>

[Overview](#-overview) &nbsp;·&nbsp; [Features](#-features) &nbsp;·&nbsp; [Architecture](#️-architecture) &nbsp;·&nbsp; [Tech Stack](#️-tech-stack) &nbsp;·&nbsp; [Getting Started](#-getting-started) &nbsp;·&nbsp; [Usage](#-usage) &nbsp;·&nbsp; [Project Structure](#-project-structure) &nbsp;·&nbsp; [Changelog](#-changelog)

</div>

---

## 📌 Overview

Safety Data Sheets (SDS) are dense, multi-section regulatory documents that contain critical chemical hazard, handling, exposure, and emergency response information. Searching through 20+ pages of small print during a lab incident or compliance review is slow and error-prone.

The **Chemical Safety RAG Assistant** is a fully local, privacy-first AI tool that lets you query SDS documents in plain English. Every answer is:

- **Grounded** — retrieved directly from your SDS PDFs, not generated from model memory
- **Cited** — each response links back to the exact source document and page number
- **Safe** — prompt-level guardrails prevent the model from speculating on critical safety data

No cloud, no API keys, no data leaving your machine.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🗄️ **Dual-Mode Knowledge Base** | Ships with a pre-indexed database of 4 chemicals. Upload your own SDS PDFs anytime to extend or replace it |
| ⚡ **Zero Re-embedding Overhead** | Upload fingerprinting ensures PDFs are only processed once — never on every chat message |
| 🧠 **Cached Embedding Model** | `all-MiniLM-L6-v2` weights are loaded once into memory via `@st.cache_resource` |
| 🛡️ **Anti-Hallucination Guardrails** | Strict system prompt prevents the model from guessing flash points, PPE specs, or first-aid steps not in the SDS |
| 📑 **Persistent Source Citations** | Every answer shows collapsible cards with the source file, page number, and extracted text — saved across the full conversation |
| 💬 **Conversational Memory** | History-aware retrieval (LCEL) allows natural follow-up questions across multiple turns |
| 🧹 **Temp File Cleanup** | Uploaded PDFs are extracted safely with `try/finally` cleanup — no leaks in `%TEMP%` |
| 🔒 **100% Local & Private** | Mistral 7B runs via Ollama on your machine. No external API calls, no telemetry |
| 🔄 **Smart Fallback** | When custom uploads are removed, the app automatically reverts to the pre-indexed database |

---

## 🏗️ Architecture

The system operates in two phases: **Ingestion** (building the vector store) and **Query** (answering questions at runtime).

### Ingestion Pipeline

```
SDS PDFs  ──►  PyPDFLoader  ──►  RecursiveCharacterTextSplitter  ──►  HuggingFace Embeddings  ──►  FAISS Index
                                  chunk_size=800, overlap=150          all-MiniLM-L6-v2
```

### Query Pipeline (per user message)

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Query Pipeline                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  👤 User Question                                                      │
│         │                                                              │
│         ▼                                                              │
│  ┌─────────────────────┐     ┌──────────────────────────────────────┐ │
│  │ History-Aware       │────►│  FAISS Similarity Search             │ │
│  │ Retriever (LCEL)    │     │  top-k = 4 chunks                    │ │
│  └─────────────────────┘     └──────────────┬───────────────────────┘ │
│         ▲                                   │                          │
│         │ Chat History                      ▼                          │
│         │                    ┌──────────────────────────────────────┐ │
│  ┌──────┴────────┐           │ Prompt Template                      │ │
│  │ Session State │           │ (Safety Guardrails + SDS Context)    │ │
│  └───────────────┘           └──────────────┬───────────────────────┘ │
│                                             │                          │
│                                             ▼                          │
│                              ┌──────────────────────────────────────┐ │
│                              │  Mistral 7B via Ollama (local)       │ │
│                              └──────────────┬───────────────────────┘ │
│                                             │                          │
│                                             ▼                          │
│                       ✅  Answer  +  📑 Source Citations              │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Dual-Mode Vector Store

```
App Startup
    │
    ├── No uploads? ──► Load vectorstore/ from disk (pre-indexed)
    │
    └── Files uploaded?
            ├── New files detected? ──► Re-index only the new batch
            └── Same files as before? ──► Reuse existing chain (no re-embedding)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|:---|:---|:---|
| **UI** | [Streamlit](https://streamlit.io/) | `>=1.32` |
| **LLM** | [Ollama](https://ollama.com/) — Mistral 7B | `mistral:latest` |
| **RAG Orchestration** | [LangChain](https://python.langchain.com/) (LCEL) | `>=0.2` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | via `langchain-huggingface` |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (CPU) | `>=1.8` |
| **PDF Parsing** | `pypdf` | `>=4.2` |
| **Language** | Python | `3.10 – 3.12` |

---

## 📦 Getting Started

### Prerequisites

Before running the app, ensure you have:

- Python `3.10` or higher installed
- [Ollama](https://ollama.com/) installed and running on your machine
- The Mistral model pulled:

  ```bash
  ollama pull mistral
  ```

> **Tip:** You can verify Ollama is running with `ollama list`. You should see `mistral:latest` in the output.

---

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Mona-Agrawall/chemical-safety-rag-assistant.git
cd chemical-safety-rag-assistant
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch the app**

```bash
streamlit run app.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## 🚀 Usage

### On First Launch

The app automatically loads a pre-indexed database from `vectorstore/` containing four industrial chemicals:
**Acetone**, **Benzene**, **Ethanol**, and **Methanol**.

The sidebar will show:
```
🗄️ Active Source: Pre-indexed SDS database (Acetone, Benzene, Ethanol, Methanol)
```

You can start asking questions immediately — no setup required.

---

### Uploading Custom SDS Documents

1. Drag and drop one or more SDS PDF files into the upload area at the top of the page
2. The app detects new files, indexes them, and switches the active knowledge base
3. The sidebar updates to confirm:
   ```
   📁 Active Source: Uploaded files: <your_file.pdf> (N chunks)
   ```
4. If you remove uploaded files, the app automatically reverts to the pre-indexed database

---

### Asking Questions

Type your question in the chat box at the bottom, or click any of the **preset questions** in the sidebar.

**Example prompts:**

```
What are the primary hazards and GHS classifications of acetone?
What glove material is recommended for handling benzene?
What is the flash point of ethanol?
What should I do if methanol is ingested?
How should acetone be stored to avoid fire risk?
```

---

### Reading Source Citations

Every answer includes a collapsible **📑 Referenced SDS Sources** section showing:
- The source PDF filename
- The exact page number
- The raw text excerpt used to generate the answer

Citations are permanently attached to each message and remain visible as you scroll through the conversation.

---

### Sidebar Controls

| Control | Action |
|:---|:---|
| **🧹 Clear Chat History** | Resets all messages and conversational memory |
| **Knowledge Base Status** | Shows the currently active data source |
| **💡 Example Questions** | Clickable preset prompts for common SDS queries |

---

## 📁 Project Structure

```
chemical-safety-rag-assistant/
│
├── 📂 data/                    # Pre-loaded SDS PDFs
│   ├── acetone_sds.pdf
│   ├── benzene_sds.pdf
│   ├── ethanol_sds.pdf
│   └── methanol_sds.pdf
│
├── 📂 vectorstore/             # Pre-built FAISS index (ready to use)
│   ├── index.faiss
│   └── index.pkl
│
├── 📄 app.py                   # Main Streamlit app — RAG chain, UI, state management
├── 📄 ingest.py                # Offline batch ingestion script for data/ → vectorstore/
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md
└── 📄 .gitignore
```

---

## ⚙️ Offline Batch Ingestion

To rebuild or extend the default vector store using PDFs in `data/`:

1. Add your SDS PDF files to the `data/` directory
2. Run the ingestion script:

   ```bash
   python ingest.py
   ```

3. The script will:
   - Load all PDFs from `data/`
   - Split them into chunks (`size=800`, `overlap=150`)
   - Generate embeddings using `all-MiniLM-L6-v2`
   - Save the updated FAISS index to `vectorstore/`

The next time you launch `app.py`, the new index will be loaded automatically.

---

## 📝 Changelog

### v2.0 — Performance & Safety Overhaul

- **Fixed: Re-embedding on every chat message** — Added upload fingerprinting; the pipeline only re-indexes when files actually change
- **Fixed: Temporary PDF file leakage** — `try/finally` blocks guarantee disk cleanup after PDF extraction
- **Fixed: Vectorstore ignored at startup** — App now auto-loads the pre-built `vectorstore/` on launch; no upload required to begin querying
- **Fixed: Missing `langchain-classic` dependency** — Added to `requirements.txt`; import fallback added for forward compatibility
- **Improved: Embedding model caching** — `@st.cache_resource` prevents PyTorch weights from being reloaded into RAM on each run
- **Improved: Chunking consistency** — `chunk_size` and `chunk_overlap` are now identical between `app.py` and `ingest.py` (`800` / `150`)
- **Improved: Anti-hallucination system prompt** — Model is explicitly prohibited from speculating on safety-critical properties absent from the SDS
- **New: Sidebar controls** — Clear Chat History button, live knowledge base status badge, and clickable sample questions
- **New: Persistent source citations** — Source cards (file + page + excerpt) are now saved with each message and remain visible throughout the session
- **New: Automatic fallback** — When uploaded files are removed, the app reverts to the pre-indexed database without requiring a page refresh

---

## 🔮 Roadmap

- [x] Multi-turn chat with conversational memory
- [x] Dynamic PDF upload via the UI
- [x] Source citations with filename and page number
- [x] Pre-indexed database with automatic startup loading
- [x] Performance-optimized pipeline (no redundant re-embedding)
- [x] Anti-hallucination safety guardrails
- [ ] Streaming LLM responses (token-by-token)
- [ ] Hybrid retrieval (MMR / BM25 + dense)
- [ ] Multi-chemical comparison queries
- [ ] Docker containerization for one-command deployment
- [ ] Support for additional LLM backends (GPT-4o, Claude)

---

## 🖼️ Demo

<img width="844" alt="Chemical Safety Assistant — Chat Interface" src="https://github.com/user-attachments/assets/f3f3c104-b15d-4780-abbf-997e558bc934"/>
<img width="887" alt="Chemical Safety Assistant — Source Citations" src="https://github.com/user-attachments/assets/0404ab4b-55ae-4c8b-88eb-b328abeab8b7"/>

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ by [Mona Agrawal](https://github.com/Mona-Agrawall)

⭐ If this project helped you, consider giving it a star!

</div>
