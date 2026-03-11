<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
<img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

# 🧪 Chemical Safety RAG Assistant

### *Intelligent Q&A over Chemical Safety Data Sheets — powered by local LLMs*

A **Retrieval-Augmented Generation (RAG)** system that lets you ask natural language questions about chemical hazards, PPE, first aid, and storage — and get grounded, cited answers from real Safety Data Sheets.

[Features](#features) · [Architecture](#architecture) · [Tech Stack](#tech-stack) · [Installation](#installation) · [Usage](#usage) · [Roadmap](#roadmap)

</div>

---

## ✨ Features

| Capability | Description |
|---|---|
| 🔍 **Semantic Search** | FAISS vector search over chunked SDS documents |
| 🤖 **Local LLM** | Mistral model via Ollama — fully offline, no API keys |
| 📄 **Source Citations** | Every answer references the source page(s) from the SDS |
| 🧠 **Smart Embeddings** | MiniLM sentence-transformer embeddings for high-quality retrieval |
| 🌐 **Web Interface** | Clean, interactive Streamlit UI |
| ⚗️ **Domain-Focused** | Optimized for chemical safety: hazards, PPE, flash points, first aid |

---

## 🏗️ Architecture

The pipeline takes a user question through vector search, retrieval, and local LLM inference to produce a grounded answer with source references.

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   👤 User Question                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────┐                                          │
│   │  Sentence        │  ← MiniLM Embeddings                   │
│   │  Transformer     │                                         │
│   └────────┬────────┘                                          │
│            │  Query Vector                                      │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │  FAISS Vector   │  ← Similarity Search                    │
│   │  Store          │                                          │
│   └────────┬────────┘                                          │
│            │  Top-K Relevant SDS Chunks                        │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │  Prompt          │  ← Context + Question Assembly          │
│   │  Template        │                                         │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │  Mistral LLM    │  ← Local Inference via Ollama           │
│   │  (Ollama)        │                                         │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ✅ Answer + Source Page References                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Ingestion Pipeline

```
📁 SDS PDFs  →  📄 Text Extraction  →  ✂️ Chunking  →  🔢 Embeddings  →  💾 FAISS Index
```

---

## 🛠️ Tech Stack

<table>
<tr>
<td><b>Layer</b></td>
<td><b>Technology</b></td>
<td><b>Purpose</b></td>
</tr>
<tr>
<td>UI</td>
<td>Streamlit</td>
<td>Web interface</td>
</tr>
<tr>
<td>Orchestration</td>
<td>LangChain</td>
<td>RAG chain, prompt templates, document loaders</td>
</tr>
<tr>
<td>Vector Store</td>
<td>FAISS</td>
<td>Fast similarity search over document embeddings</td>
</tr>
<tr>
<td>Embeddings</td>
<td>Sentence Transformers (MiniLM)</td>
<td>Dense vector representations of text</td>
</tr>
<tr>
<td>LLM</td>
<td>Mistral via Ollama</td>
<td>Local language model inference</td>
</tr>
<tr>
<td>Language</td>
<td>Python 3.10+</td>
<td>Core application language</td>
</tr>
</table>

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mona-Agrawall/chemical-safety-rag-assistant.git
cd chemical-safety-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull the model

Download Ollama from **[ollama.com](https://ollama.com)**, then pull Mistral:

```bash
ollama pull mistral
```

### 5. Add your SDS documents

Place your SDS PDF files in the `data/` directory:

```
data/
├── acetone_sds.pdf
├── benzene_sds.pdf
└── ethanol_sds.pdf
```

### 6. Run the ingestion pipeline

```bash
python ingest.py
```

This processes your PDFs, generates embeddings, and saves the FAISS vector store to `vectorstore/`.

---

## 🚀 Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to:

```
http://localhost:8501
```

### 💬 Example Questions

```
❓ What are the hazards of acetone?
❓ What protective equipment is required for benzene?
❓ What is the flash point of ethanol?
❓ What should I do if methanol is ingested?
❓ How should I store hydrogen peroxide?
❓ What are the first aid measures for sulfuric acid exposure?
```

---

## 📁 Project Structure

```
chemical-safety-rag-assistant/
│
├── 📂 data/               # SDS PDF documents
│   └── *.pdf
│
├── 📂 vectorstore/        # FAISS vector index (auto-generated)
│   ├── index.faiss
│   └── index.pkl
│
├── 📄 app.py              # Streamlit web application & RAG chain
├── 📄 ingest.py           # PDF loader, chunker & embedding generator
├── 📄 requirements.txt    # Python dependencies
└── 📄 .gitignore
```

---

## 🔮 Roadmap

- [ ] 💬 Multi-turn chat interface with conversation memory
- [ ] 📤 Dynamic PDF upload via the UI
- [ ] ⚡ Streaming LLM responses
- [ ] 📊 Improved retrieval ranking (MMR / hybrid search)
- [ ] 🐳 Docker containerization
- [ ] 🔧 Support for additional LLM backends (GPT-4, Claude)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [Mona Agrawal](https://github.com/Mona-Agrawall)

⭐ *If you found this useful, consider giving it a star!*

</div>
