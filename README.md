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

[Features](#-features) · [Architecture](#️-architecture) · [Tech Stack](#️-tech-stack) · [Installation](#-installation) · [Usage](#-usage) · [Roadmap](#-roadmap)

</div>

---

## ✨ Features

| Capability | Description |
|---|---|
| 🔍 **Semantic Search** | FAISS vector search over chunked SDS documents |
| 🤖 **Local LLM** | Mistral model via Ollama — fully offline, no API keys |
| 📄 **Source Citations** | Every answer references the original PDF filename and page number |
| 🧠 **Smart Embeddings** | MiniLM sentence-transformer embeddings for high-quality retrieval |
| 🌐 **Web Interface** | Clean, interactive Streamlit UI |
| ⚗️ **Domain-Focused** | Optimized for chemical safety: hazards, PPE, flash points, first aid |
| 💬 **Conversation Memory** | Multi-turn chat with full conversation history |
| 📤 **Dynamic PDF Upload** | Upload SDS sheets directly through the UI — no ingestion script needed |

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

### 5. Run the app

```bash
streamlit run app.py
```

Upload your SDS PDFs directly through the UI — no separate ingestion step needed.

---

## 🚀 Usage

Open your browser and navigate to:

```
http://localhost:8501
```

1. **Upload** one or more SDS PDF files using the file uploader
2. **Wait** for the documents to be processed (embeddings are generated automatically)
3. **Ask** any chemical safety question in the chat input

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
├── 📂 data/               # (Optional) SDS PDF documents for batch ingestion
│   └── *.pdf
│
├── 📄 app.py              # Streamlit web application & RAG chain
├── 📄 ingest.py           # (Optional) Offline PDF loader, chunker & embedding generator
├── 📄 requirements.txt    # Python dependencies
└── 📄 .gitignore
```

---

## 🔮 Roadmap

- [x] 💬 Multi-turn chat interface with conversation memory
- [x] 📤 Dynamic PDF upload via the UI
- [x] 📄 Source citations with original filename and page number
- [ ] ⚡ Streaming LLM responses
- [ ] 📊 Improved retrieval ranking (MMR / hybrid search)
- [ ] 🐳 Docker containerization
- [ ] 🔧 Support for additional LLM backends (GPT-4, Claude)

---

## 🖼️ Demo

<img width="815" height="1034" alt="image" src="https://github.com/user-attachments/assets/f212835b-0ecb-447e-a706-49169a28d19b" />

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [Mona Agrawal](https://github.com/Mona-Agrawall)

⭐ *If you found this useful, consider giving it a star!*

</div>