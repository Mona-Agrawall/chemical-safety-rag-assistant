import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chains import with robust compatibility fallback
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
    from langchain_classic.chains.retrieval import create_retrieval_chain

# ─────────────────────────────────────────────
# Constants & Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"
DATA_PATH = BASE_DIR / "data"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_CHUNKS = 5          # Retrieve 5 chunks for richer context
MAX_HISTORY_TURNS = 6     # Keep only last 6 turns (12 messages) to stay within LLM context window
OLLAMA_MODEL = "mistral"

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chemical Safety RAG Assistant",
    page_icon="⚗️",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom Styling
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">

<style>

/* ── Global reset & font ─────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'EB Garamond', Georgia, serif !important;
    background-color: #F8F7F5 !important;
    color: #1A1A1A !important;
}

/* ── Main content area ───────────────────────────────────────── */
.main .block-container {
    background-color: #F8F7F5 !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 900px !important;
}

/* ── App title ───────────────────────────────────────────────── */
h1 {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 2.4rem !important;
    font-weight: 600 !important;
    color: #0F0F0F !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
    margin-bottom: 0.3rem !important;
}

h2, h3 {
    font-family: 'EB Garamond', Georgia, serif !important;
    color: #1A1A1A !important;
    font-weight: 600 !important;
}

/* ── Subtitle / body text ────────────────────────────────────── */
p, .stMarkdown p {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 1.1rem !important;
    color: #3D3D3D !important;
    line-height: 1.7 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E8E4DF !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #0F0F0F !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.95rem !important;
    color: #5A5A5A !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    color: #888888 !important;
}

/* ── Sidebar buttons (preset questions) ─────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background-color: #F5F3F0 !important;
    color: #2A2A2A !important;
    border: 1px solid #E2DDD8 !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 0.8rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
    border-color: #1A1A1A !important;
    transform: translateX(2px) !important;
}

/* ── Clear history button ─────────────────────────────────────── */
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.18s ease !important;
}

[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background-color: #3A3A3A !important;
    transform: none !important;
}

/* ── Dividers ────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #E8E4DF !important;
    margin: 1.2rem 0 !important;
}

/* ── File uploader ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1.5px dashed #C8C0B8 !important;
    border-radius: 10px !important;
    padding: 1.2rem !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #1A1A1A !important;
}

[data-testid="stFileUploader"] label {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 1rem !important;
    color: #3D3D3D !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] .stCaption {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    color: #8A8A8A !important;
}

/* ── Chat messages ───────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    margin-bottom: 0.8rem !important;
    padding: 1rem 1.2rem !important;
    font-family: 'EB Garamond', Georgia, serif !important;
}

/* User message */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #EFEFED !important;
    border: 1px solid #E4E0DA !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #FFFFFF !important;
    border: 1px solid #E8E4DF !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}

[data-testid="stChatMessage"] p {
    font-size: 1.05rem !important;
    line-height: 1.75 !important;
    color: #1A1A1A !important;
}

/* Chat avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background-color: #8B7355 !important;
    color: #FFFFFF !important;
}

/* ── Chat input bar ──────────────────────────────────────────── */
[data-testid="stChatInputContainer"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #D8D2CB !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stChatInputContainer"]:focus-within {
    border-color: #1A1A1A !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.1) !important;
}

[data-testid="stChatInputContainer"] textarea {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 1.05rem !important;
    color: #1A1A1A !important;
    background-color: transparent !important;
}

[data-testid="stChatInputContainer"] textarea::placeholder {
    color: #ABABAB !important;
    font-style: italic !important;
}

/* Chat send button */
[data-testid="stChatInputSubmitButton"] > button {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
    border-radius: 7px !important;
    border: none !important;
    transition: background-color 0.18s ease !important;
}

[data-testid="stChatInputSubmitButton"] > button:hover {
    background-color: #3A3A3A !important;
}

/* ── Expanders (source citations) ────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E8E4DF !important;
    border-radius: 8px !important;
    margin-top: 0.6rem !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #5A5A5A !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 0.8rem !important;
}

[data-testid="stExpander"] summary:hover {
    color: #1A1A1A !important;
}

[data-testid="stExpander"] .stText,
[data-testid="stExpander"] pre {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    color: #5A5A5A !important;
    background-color: #F8F7F5 !important;
    border-radius: 6px !important;
    padding: 0.6rem 0.8rem !important;
    line-height: 1.55 !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

.stCaption, small {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    color: #7A7A7A !important;
    font-weight: 500 !important;
}

/* ── Alerts (success / info / warning / error) ───────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    border-left-width: 3px !important;
}

/* Success */
[data-testid="stAlert"][data-baseweb="notification"][kind="positive"] {
    background-color: #F2FAF5 !important;
    border-left-color: #2D7A55 !important;
    color: #1A4830 !important;
}

/* Info */
[data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
    background-color: #F5F7FA !important;
    border-left-color: #5A7A9A !important;
    color: #2A3D50 !important;
}

/* Warning */
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {
    background-color: #FDF8F0 !important;
    border-left-color: #A07840 !important;
    color: #5A4020 !important;
}

/* Error */
[data-testid="stAlert"][data-baseweb="notification"][kind="negative"] {
    background-color: #FDF2F2 !important;
    border-left-color: #A03535 !important;
    color: #5A1A1A !important;
}

/* ── Spinner ─────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    color: #6A6A6A !important;
    font-style: italic !important;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
::-webkit-scrollbar-track {
    background: #F0EDE9;
}
::-webkit-scrollbar-thumb {
    background: #C8C0B8;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #9A9490;
}

/* ── Tooltip / help icon ─────────────────────────────────────── */
[data-testid="stTooltipIcon"] {
    color: #ABABAB !important;
}

/* ── Selection highlight ─────────────────────────────────────── */
::selection {
    background-color: #C8B89A !important;
    color: #1A1A1A !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Cached Resources (load once per server session)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load MiniLM sentence-transformer weights once into memory."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOllama:
    """Create a single shared Ollama client (not re-instantiated on chain rebuild)."""
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


def check_ollama() -> tuple[bool, str]:
    """
    Probe Ollama with a trivial request to confirm it's reachable before
    any user interaction.  Returns (ok: bool, error_message: str).
    """
    try:
        llm = get_llm()
        llm.invoke("ping")
        return True, ""
    except Exception as e:
        err = str(e)
        if "Connection refused" in err or "ConnectError" in err or "connect" in err.lower():
            return False, (
                "Cannot connect to Ollama. Make sure it is running.\n\n"
                "**Start Ollama:** open a terminal and run `ollama serve`\n\n"
                f"**Then pull the model:** `ollama pull {OLLAMA_MODEL}`"
            )
        return False, f"Ollama error: {err}"


# ─────────────────────────────────────────────
# PDF Processing Utilities
# ─────────────────────────────────────────────
def _file_signature(files) -> str:
    """
    Produce a stable hash from (name, size) pairs so we only re-index
    when the actual file set changes, not on every Streamlit rerun.
    """
    key = str(sorted((f.name, f.size) for f in files))
    return hashlib.md5(key.encode()).hexdigest()


def process_pdfs(files) -> list:
    """
    Extract LangChain Documents from uploaded PDF files.
    Guarantees deletion of temporary disk copies via try/finally.
    """
    documents = []
    for file in files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.seek(0)
                tmp.write(file.read())
                tmp_path = tmp.name
            docs = PyPDFLoader(tmp_path).load()
            for doc in docs:
                doc.metadata["source"] = file.name
            documents.extend(docs)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    return documents


def split_documents(documents: list) -> list:
    """Split documents into overlapping chunks using shared constants."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(documents)


def create_vectorstore(chunks: list) -> FAISS:
    """Build an in-memory FAISS index from document chunks."""
    return FAISS.from_documents(chunks, get_embeddings())


def load_default_vectorstore() -> FAISS | None:
    """Load the pre-built FAISS index from disk if it exists."""
    faiss_file = VECTORSTORE_PATH / "index.faiss"
    if VECTORSTORE_PATH.exists() and faiss_file.exists():
        return FAISS.load_local(
            str(VECTORSTORE_PATH),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return None


# ─────────────────────────────────────────────
# RAG Chain Construction
# ─────────────────────────────────────────────
def create_chain(vectorstore: FAISS):
    """
    Build the full conversational RAG chain:
      1. history_aware_retriever  – rewrites ambiguous follow-up questions
      2. stuff_documents_chain    – fills the safety-focused QA prompt
      3. retrieval_chain          – wires them together

    Uses the module-level cached LLM so Ollama is not re-connected on
    every vectorstore switch.
    """
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})

    # Rewrite follow-up questions to be self-contained
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rewrite the question "
         "as a fully self-contained query that does NOT rely on prior context. "
         "Return it unchanged if it is already self-contained. Do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Chemical-safety-specific QA prompt with strict grounding rules
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a specialist Chemical Safety Assistant. Your answers must be "
         "based SOLELY on the Safety Data Sheet (SDS) context provided below.\n\n"
         "RULES — follow these exactly:\n"
         "1. Do NOT speculate, extrapolate, or use general chemistry knowledge to "
         "fill gaps. Chemical properties (flash points, exposure limits, PPE "
         "materials, first-aid steps) MUST come from the retrieved text.\n"
         "2. If the context does not contain the requested information, say EXACTLY: "
         "'The provided Safety Data Sheets do not contain this information.'\n"
         "3. Cite the source document name and page number at the end of every "
         "factual statement, e.g. *(acetone_sds.pdf – p. 4)*.\n"
         "4. Use clear, plain language suitable for a lab technician or first responder.\n\n"
         "SDS Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


# ─────────────────────────────────────────────
# History Windowing
# ─────────────────────────────────────────────
def get_windowed_history() -> list:
    """
    Return only the most recent MAX_HISTORY_TURNS turn-pairs from the
    LangChain chat_history list to avoid overflowing the LLM context window.
    Each turn = 1 HumanMessage + 1 AIMessage = 2 list items.
    """
    history = st.session_state.get("chat_history", [])
    max_messages = MAX_HISTORY_TURNS * 2
    return history[-max_messages:] if len(history) > max_messages else history


# ─────────────────────────────────────────────
# Source Deduplication
# ─────────────────────────────────────────────
def deduplicate_sources(context_docs: list) -> list:
    """
    Remove duplicate chunks by (source, page) so the citation panel
    doesn't show the same page excerpt multiple times.
    """
    seen = set()
    unique = []
    for doc in context_docs:
        src = doc.metadata.get("source", "Unknown")
        raw = doc.metadata.get("page", 0)
        page = (raw + 1) if isinstance(raw, int) else raw
        key = (src, page)
        if key not in seen:
            seen.add(key)
            unique.append((src, page, doc.page_content.strip()))
    return unique


# ─────────────────────────────────────────────
# Session State Helpers
# ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "messages": [],
        "chat_history": [],
        "current_upload_sig": None,
        "active_source": "Initializing…",
        "ollama_ok": None,   # None = not checked yet
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_chain_from_default():
    """Attempt to load the pre-built vectorstore and wire the chain."""
    vs = load_default_vectorstore()
    if vs:
        st.session_state.chain = create_chain(vs)
        st.session_state.active_source = (
            "Pre-indexed SDS database (Acetone, Benzene, Ethanol, Methanol)"
        )
    else:
        st.session_state.pop("chain", None)
        st.session_state.active_source = "No database found. Upload SDS PDFs to begin."


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Controls")

        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        st.divider()

        # Ollama status
        st.subheader("🤖 Ollama Status")
        if st.session_state.ollama_ok is True:
            st.success(f"✅ Connected — `{OLLAMA_MODEL}` ready")
        elif st.session_state.ollama_ok is False:
            st.error("❌ Ollama unreachable")
        else:
            st.info("⏳ Checking…")

        st.divider()

        # Knowledge base status
        st.subheader("🗄️ Knowledge Base")
        src = st.session_state.get("active_source", "")
        if "Uploaded" in src:
            st.info(f"📁 {src}")
        elif "Pre-indexed" in src:
            st.success(f"🗄️ {src}")
        elif "Initializing" in src:
            st.info(f"⏳ {src}")
        else:
            st.warning(f"⚠️ {src}")

        # History window indicator
        total_turns = len(st.session_state.chat_history) // 2
        st.caption(
            f"Conversation: {total_turns} turns "
            f"(window: last {min(total_turns, MAX_HISTORY_TURNS)} sent to LLM)"
        )

        st.divider()

        # Preset questions
        st.subheader("💡 Example Questions")
        presets = [
            "What are the primary hazards of acetone?",
            "What PPE is required when handling benzene?",
            "What is the flash point of ethanol?",
            "What first-aid measures apply to methanol ingestion?",
            "How should acetone be stored safely?",
        ]
        for q in presets:
            if st.button(f"🔍 {q}", key=f"preset_{q}", use_container_width=True):
                st.session_state.preset_query = q

        st.divider()
        st.caption(
            "🔒 Fully offline · No API keys required\n\n"
            f"Model: **{OLLAMA_MODEL}** · Embeddings: **all-MiniLM-L6-v2**"
        )


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
init_session_state()

# ── Ollama health check (runs once per session) ──────────────────────────
if st.session_state.ollama_ok is None:
    with st.spinner("Checking Ollama connection…"):
        ok, err_msg = check_ollama()
    st.session_state.ollama_ok = ok
    if not ok:
        st.error(f"⚠️ {err_msg}")

render_sidebar()

# ── Page header ──────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 0.2rem;">
    <span style="
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8B7355;
    ">Chemical Safety · RAG Assistant</span>
</div>
<h1 style="
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #0F0F0F;
    letter-spacing: -0.025em;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
">⚗️ Ask Your Safety Data Sheets</h1>
<p style="
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1.1rem;
    color: #6A6A6A;
    margin: 0 0 1.5rem 0;
    line-height: 1.6;
    font-style: italic;
">Grounded answers on chemical hazards, PPE, first aid, and storage —
cited directly from your SDS documents.</p>
<div style="width: 48px; height: 2px; background-color: #8B7355; border-radius: 2px; margin-bottom: 1.5rem;"></div>
""", unsafe_allow_html=True)

# ── File uploader ─────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📎  Upload Safety Data Sheets (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Uploads replace the active knowledge base for this session. Remove files to revert to the pre-indexed database.",
)

# ── Vectorstore / chain state management ─────────────────────────────────
new_sig = _file_signature(uploaded_files) if uploaded_files else None

if uploaded_files and new_sig != st.session_state.current_upload_sig:
    # New or changed files — reindex
    with st.spinner(f"Indexing {len(uploaded_files)} file(s)…"):
        try:
            docs = process_pdfs(uploaded_files)
            if docs:
                chunks = split_documents(docs)
                vs = create_vectorstore(chunks)
                st.session_state.chain = create_chain(vs)
                names = ", ".join(f.name for f in uploaded_files)
                st.session_state.active_source = (
                    f"Uploaded: {names} — {len(chunks)} chunks"
                )
                st.session_state.current_upload_sig = new_sig
                st.toast(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")
            else:
                st.error("❌ No text could be extracted from the uploaded PDFs.")
        except Exception as exc:
            st.error(f"❌ Indexing failed: {exc}")

elif not uploaded_files and st.session_state.current_upload_sig is not None:
    # Files removed — revert to default
    st.session_state.current_upload_sig = None
    with st.spinner("Loading pre-indexed database…"):
        load_chain_from_default()
    st.toast("Reverted to the pre-indexed chemical database.")

elif "chain" not in st.session_state:
    # First load — pull default vectorstore
    with st.spinner("Loading pre-indexed SDS database…"):
        load_chain_from_default()

st.divider()

# ── Chat history display ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📑 Referenced SDS Sources", expanded=False):
                for src in msg["sources"]:
                    st.caption(
                        f"**{src['source']}** · Page {src['page']}"
                    )
                    st.text(src["content"])
                    st.divider()

# ── Chat input ────────────────────────────────────────────────────────────
preset = st.session_state.pop("preset_query", None)
user_input = st.chat_input(
    "Ask a chemical safety question…",
    disabled=not st.session_state.get("ollama_ok", False),
)
query = preset or user_input

# ── Input validation ──────────────────────────────────────────────────────
if query and not query.strip():
    query = None  # ignore whitespace-only input

if query:
    if "chain" not in st.session_state:
        st.warning("⚠️ Please wait for the database to finish loading, or upload SDS PDFs.")
    elif not st.session_state.ollama_ok:
        st.error("❌ Ollama is not running. Please start it and refresh the page.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching SDS documents and generating answer…"):
                    result = st.session_state.chain.invoke({
                        "input": query,
                        "chat_history": get_windowed_history(),
                    })

                answer = result.get("answer", "").strip()
                if not answer:
                    answer = "I was unable to generate an answer. Please try rephrasing your question."

                st.markdown(answer)

                # Deduplicated source citations
                raw_context = result.get("context", [])
                unique_sources = deduplicate_sources(raw_context)

                if unique_sources:
                    with st.expander("📑 Referenced SDS Sources", expanded=False):
                        for src_name, page_num, snippet in unique_sources:
                            st.caption(f"**{src_name}** · Page {page_num}")
                            st.text(snippet)
                            st.divider()

                # Persist to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": [
                        {"source": s, "page": p, "content": c}
                        for s, p, c in unique_sources
                    ],
                })
                st.session_state.chat_history.extend([
                    HumanMessage(content=query),
                    AIMessage(content=answer),
                ])

            except Exception as exc:
                err = str(exc)
                if "Connection refused" in err or "ConnectError" in err:
                    st.error(
                        "❌ Lost connection to Ollama mid-session. "
                        "Please ensure `ollama serve` is running and refresh the page."
                    )
                    st.session_state.ollama_ok = False
                else:
                    st.error(f"❌ Error generating response: {exc}")