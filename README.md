<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
<img src="https://img.shields.io/badge/Ollama-Mistral%207B-000000?style=for-the-badge&logo=ollama&logoColor=white"/>
<img src="https://img.shields.io/badge/FAISS-Vector%20Store-00599C?style=for-the-badge&logo=meta&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

# 🧪 Chemical Safety RAG Assistant

### *Grounded, Private Q&A over Chemical Safety Data Sheets (SDS) — Powered by Local LLMs*

A robust, enterprise-ready **Retrieval-Augmented Generation (RAG)** system engineered to answer complex questions about chemical hazards, personal protective equipment (PPE), first aid procedures, and storage protocols directly from authentic Safety Data Sheets.

[Key Features](#-key-features) · [System Architecture](#️-system-architecture) · [Tech Stack](#️-tech-stack) · [Getting Started](#-getting-started) · [Project Structure](#-project-structure) · [Demos](#-demo) · [License](#-license)

</div>

---

## 📌 Overview

Safety Data Sheets (SDSs) are dense, multi-page regulatory documents containing critical safety, handling, and toxicological information across 16 standardized sections. In emergency situations or lab operations, searching through 20+ pages of small print for flash points, glove breakthrough times, or specific first-aid protocols is slow and prone to human error.

The **Chemical Safety RAG Assistant** provides an offline, private, and mathematically grounded assistant that extracts answers directly from SDS documents with **exact filename and page citations**. Because chemical safety is high-stakes, the system incorporates strict anti-hallucination guardrails and runs entirely locally — ensuring zero cloud dependencies, zero data leakage, and zero API costs.

---

## ✨ Key Features

* **🗄️ Dual-Mode Knowledge Base**:
  * **Out-of-the-Box Database**: Pre-indexed with industrial chemicals (**Acetone**, **Benzene**, **Ethanol**, and **Methanol**). You can start querying immediately upon launch.
  * **Dynamic PDF Ingestion**: Upload custom SDS PDFs on the fly through the Streamlit UI to instantly augment or replace the active vector space.
* **⚡ State-Managed & Cached Pipeline**:
  * **Zero Re-embedding Penalty**: Utilizes upload fingerprinting so user interactions never trigger duplicate embedding cycles.
  * **Model Caching (`@st.cache_resource`)**: Loads Hugging Face transformer weights once into memory, preventing memory churn.
* **🛡️ Anti-Hallucination Chemical Guardrails**:
  * Specially tailored prompt constraints prohibit the model from speculating or generalizing critical safety data (e.g., flash points, incompatible chemicals, first-aid measures).
  * If a requested detail is missing from the SDS, the assistant explicitly states: *"The provided Safety Data Sheets do not contain this information."*
* **📑 Persistent Source Citations**:
  * Every answer includes collapsible reference cards displaying the exact source document, page number, and relevant text snippet.
  * Citations remain permanently attached to earlier messages throughout multi-turn discussions.
* **💬 Conversational Memory (LCEL)**:
  * Multi-turn chat enabled by LangChain's `history_aware_retriever`, allowing natural follow-up questions (e.g., *"What PPE is needed for it?"*).
* **🔒 100% Offline & Private**:
  * Local LLM inference via **Ollama (Mistral 7B)** and local dense embeddings via **Sentence-Transformers (`all-MiniLM-L6-v2`)**.
  * No external API calls, tracking, or subscription tokens required.
* **🧹 Clean Temporary Resource Management**:
  * Uploaded PDF buffers are processed safely with strict `try...finally` disk cleanup, preventing `%TEMP%` leakage.

---

## 🏗️ System Architecture

```
                                ┌──────────────────────────────────────┐
                                │   Safety Data Sheets (data/*.pdf)    │
                                └──────────────────┬───────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ PyPDFLoader + CharacterTextSplitter  │
                                │ (Chunk Size: 800, Overlap: 150)      │
                                └──────────────────┬───────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ Sentence-Transformers: MiniLM-L6-v2  │
                                └──────────────────┬───────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │       FAISS Vector Database          │
                                └──────────────────┬───────────────────┘
                                                   │
┌─────────────────────────┐                        │
│ 👤 User Prompt          │                        │
└───────────┬─────────────┘                        │
            │                                      │
            ▼                                      ▼
┌─────────────────────────┐               ┌──────────────────────────────────────┐
│ History-Aware Retriever │ ────────────► │ Similarity Search (Top-k = 4 Chunks) │
└─────────────────────────┘               └──────────────────┬───────────────────┘
                                                             │
                                                             ▼
                                          ┌──────────────────────────────────────┐
                                          │ Context + Guardrail Safety Template  │
                                          └──────────────────┬───────────────────┘
                                                             │
                                                             ▼
                                          ┌──────────────────────────────────────┐
                                          │     Local Mistral 7B (Ollama)        │
                                          └──────────────────┬───────────────────┘
                                                             │
                                                             ▼
                                          ┌──────────────────────────────────────┐
                                          │ ✅ Grounded Answer + Source Cards     │
                                          └──────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|:---|:---|:---|
| **User Interface** | [Streamlit](https://streamlit.io/) | Interactive web UI with sidebar database status, preset questions, and history management |
| **LLM Inference** | [Ollama](https://ollama.com/) (Mistral 7B) | Local, quantized open-source language model running with 0 external network requests |
| **RAG Orchestration** | [LangChain](https://www.langchain.com/) (LCEL) | History-aware retrieval chains and document formatting |
| **Embeddings** | [HuggingFace](https://huggingface.co/) / `sentence-transformers` | `all-MiniLM-L6-v2` dense 384-dimensional semantic embeddings |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (CPU) | High-speed in-memory vector index with local disk persistence |
| **Document Parsing** | `pypdf` | Automated text extraction from regulatory PDF documents |

---

## 📦 Getting Started

### Prerequisites

1. **Python**: Version `3.10` to `3.12` installed.
2. **Ollama**: Download and install from [ollama.com](https://ollama.com).
3. **Pull Mistral Model**:
   ```bash
   ollama pull mistral
   ```
   *(Ensure Ollama is running in the background before launching the app)*

---

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/Mona-Agrawall/chemical-safety-rag-assistant.git
cd chemical-safety-rag-assistant
```

#### 2. Create & Activate a Virtual Environment

* **Windows (PowerShell / Command Prompt)**:
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```

* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Application
```bash
streamlit run app.py
```

The app will launch at: **[http://localhost:8501](http://localhost:8501)**

---

## 🚀 Usage Guide

1. **Default Database**:
   * On startup, the assistant automatically loads the pre-indexed database (`vectorstore/`) containing **Acetone, Benzene, Ethanol, and Methanol**.
   * The sidebar will show `🗄️ Active Source: Pre-indexed SDS database`.
2. **Upload Custom SDS Files**:
   * Drag and drop any new SDS PDF files into the upload box.
   * The app will index them automatically, update the active knowledge source, and switch queries to your uploaded documents.
3. **Ask Chemical Safety Questions**:
   * Type questions in the chat box or click any of the preset sample questions in the sidebar.
4. **Inspect Source Citations**:
   * Expand the **📑 Referenced SDS Sources** toggle under any assistant response to review the exact page text and file source.
5. **Clear History**:
   * Click **🧹 Clear Chat History** in the sidebar to reset conversational memory at any time.

### 💡 Example Prompts

```text
❓ What are the primary hazards and GHS classifications of acetone?
❓ What specific glove material is recommended for handling benzene?
❓ What is the flash point and boiling point of ethanol?
❓ If methanol is ingested, what immediate first-aid steps should be taken?
❓ How should acetone be stored to prevent fire hazards?
```

---

## 📁 Project Structure

```text
chemical-safety-rag-assistant/
│
├── 📂 data/                    # Sample SDS documents (Acetone, Benzene, Ethanol, Methanol)
│   ├── acetone_sds.pdf
│   ├── benzene_sds.pdf
│   ├── ethanol_sds.pdf
│   └── methanol_sds.pdf
│
├── 📂 vectorstore/             # Pre-built FAISS vector store
│   ├── index.faiss
│   └── index.pkl
│
├── 📄 app.py                   # Main Streamlit web application & cached RAG chain
├── 📄 ingest.py                # Standalone script for batch-indexing PDFs into vectorstore/
├── 📄 requirements.txt         # Pinned Python dependencies
├── 📄 README.md                # Project documentation
└── 📄 .gitignore
```

---

## ⚙️ Offline Batch Ingestion (`ingest.py`)

If you want to add a batch of PDFs to the repository's permanent vector store on disk:

1. Place your new SDS PDF files in the `data/` folder.
2. Run the ingestion script:
   ```bash
   python ingest.py
   ```
3. The script will extract text, chunk with `chunk_size=800, overlap=150`, generate embeddings, and save the updated index to `vectorstore/`.

---

## 🖼️ Demo

<img width="844" height="839" alt="Chat Demo" src="https://github.com/user-attachments/assets/f3f3c104-b15d-4780-abbf-997e558bc934" />
<img width="887" height="1102" alt="Source Inspection Demo" src="https://github.com/user-attachments/assets/0404ab4b-55ae-4c8b-88eb-b328abeab8b7" />

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [Mona Agrawal](https://github.com/Mona-Agrawall)

⭐ *If you found this project helpful, please give it a star on GitHub!*

</div>
