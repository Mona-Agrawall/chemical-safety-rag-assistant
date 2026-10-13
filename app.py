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

# -------------------------
# Paths & Settings
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"
DATA_PATH = BASE_DIR / "data"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Chemical Safety RAG Assistant",
    page_icon="⚗️",
    layout="wide"
)

# -------------------------
# Resource Caching
# -------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Cache the sentence-transformers model so weights aren't reloaded repeatedly."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# -------------------------
# Document Ingestion Utilities
# -------------------------
def process_pdfs(files):
    """Extract documents from uploaded PDFs and guarantee cleanup of temporary files."""
    documents = []

    for file in files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                file.seek(0)
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

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


def split_documents(documents):
    """Split documents using standardized chunk size and overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    """Create a FAISS vector store from chunked documents."""
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def load_default_vectorstore():
    """Load pre-indexed FAISS database from disk if available."""
    if VECTORSTORE_PATH.exists() and (VECTORSTORE_PATH / "index.faiss").exists():
        embeddings = get_embeddings()
        return FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


# -------------------------
# RAG Chain Construction
# -------------------------
def create_chain(vectorstore):
    """Build the conversational RAG chain with anti-hallucination guardrails."""
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOllama(
        model="mistral",
        temperature=0
    )

    # 1. Contextualize Question Prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might "
         "reference context in the chat history, formulate a standalone "
         "question which can be understood without the chat history. "
         "Do NOT answer the question, just reformulate it if needed "
         "and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Strict Chemical Safety QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert Chemical Safety Assistant. Use the following Safety Data Sheet (SDS) "
         "context to answer the user's question accurately and concisely.\n\n"
         "CRITICAL SAFETY GUIDELINES:\n"
         "1. Ground your answers strictly on the provided SDS context. Do NOT guess, extrapolate, or "
         "assume chemical properties, exposure limits, PPE glove materials, or first aid measures.\n"
         "2. If the provided context does not contain the answer, explicitly state: "
         "'The provided Safety Data Sheets do not contain this information.'\n"
         "3. Always cite the source SDS document and page number for safety recommendations.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )


# -------------------------
# Sidebar Controls & Information
# -------------------------
with st.sidebar:
    st.header("⚙️ Controls & Database")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.subheader("Knowledge Base Status")
    active_source = st.session_state.get("active_source", "Not initialized")
    if "Uploaded" in active_source:
        st.info(f"📁 **Active Source:**\n{active_source}")
    elif "Pre-indexed" in active_source:
        st.success(f"🗄️ **Active Source:**\n{active_source}")
    else:
        st.warning(f"⚠️ {active_source}")

    st.divider()
    st.subheader("💡 Example Questions")
    sample_queries = [
        "What are the primary hazards of acetone?",
        "What PPE is required when handling benzene?",
        "What is the flash point of ethanol?",
        "What first-aid measures apply to methanol ingestion?",
        "How should acetone be stored?"
    ]
    for sq in sample_queries:
        if st.button(f"🔍 {sq}", key=f"sq_{sq}", use_container_width=True):
            st.session_state.preset_query = sq

    st.divider()
    st.caption("🤖 Powered by local **Mistral (Ollama)** & **all-MiniLM-L6-v2** embeddings.")


# -------------------------
# Header & File Upload
# -------------------------
st.title("⚗️ Chemical Safety RAG Assistant")
st.markdown(
    "Ask natural-language questions about chemical hazards, exposure limits, PPE, "
    "storage, and first aid — grounded directly in Safety Data Sheets (SDS)."
)

uploaded_files = st.file_uploader(
    "Upload custom Safety Data Sheets (PDF) to augment or replace the database:",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------
# State Synchronization (Prevent Re-embedding)
# -------------------------
upload_signature = [(f.name, f.size) for f in uploaded_files] if uploaded_files else None

if "current_upload_signature" not in st.session_state:
    st.session_state.current_upload_signature = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Scenario A: New files uploaded or files changed
if uploaded_files and st.session_state.current_upload_signature != upload_signature:
    with st.spinner("Processing and indexing uploaded documents..."):
        try:
            docs = process_pdfs(uploaded_files)
            if docs:
                chunks = split_documents(docs)
                vs = create_vectorstore(chunks)
                st.session_state.chain = create_chain(vs)
                file_names = ", ".join([f.name for f in uploaded_files])
                st.session_state.active_source = f"Uploaded files: {file_names} ({len(chunks)} chunks)"
                st.session_state.current_upload_signature = upload_signature
                st.toast("✅ Custom documents indexed successfully!")
            else:
                st.error("❌ Could not extract text from the uploaded PDFs.")
        except Exception as e:
            st.error(f"❌ Error processing documents: {e}")

# Scenario B: Custom files removed -> Fall back to pre-indexed vectorstore
elif not uploaded_files and st.session_state.current_upload_signature is not None:
    st.session_state.current_upload_signature = None
    default_vs = load_default_vectorstore()
    if default_vs:
        st.session_state.chain = create_chain(default_vs)
        st.session_state.active_source = "Pre-indexed SDS database (Acetone, Benzene, Ethanol, Methanol)"
        st.toast("Reverted to pre-indexed default chemical database.")
    else:
        st.session_state.pop("chain", None)
        st.session_state.active_source = "No database available"

# Scenario C: Initial application load with no custom uploads
elif "chain" not in st.session_state and not uploaded_files:
    default_vs = load_default_vectorstore()
    if default_vs:
        st.session_state.chain = create_chain(default_vs)
        st.session_state.active_source = "Pre-indexed SDS database (Acetone, Benzene, Ethanol, Methanol)"
    else:
        st.session_state.active_source = "No database loaded. Please upload SDS PDFs."


st.divider()

# -------------------------
# Display Chat History with Persistent Citations
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📑 Referenced SDS Sources"):
                for src in msg["sources"]:
                    st.caption(f"**Source:** `{src['source']}` | **Page:** `{src['page']}`")
                    st.text(src["content"])

# -------------------------
# Chat Input & Response Generation
# -------------------------
preset_input = st.session_state.pop("preset_query", None)
chat_input_val = st.chat_input("Ask a chemical safety question (e.g., 'What protective equipment is required for benzene?')")
query_to_run = preset_input or chat_input_val

if query_to_run:
    if "chain" not in st.session_state:
        st.warning("⚠️ Please wait for the database to load or upload SDS PDF files.")
    else:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.markdown(query_to_run)

        # Generate assistant answer
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing Safety Data Sheets with Mistral..."):
                    result = st.session_state.chain.invoke({
                        "input": query_to_run,
                        "chat_history": st.session_state.chat_history
                    })

                answer = result.get("answer", "")
                st.markdown(answer)

                # Format and display sources
                formatted_sources = []
                context_docs = result.get("context", [])
                if context_docs:
                    with st.expander("📑 Referenced SDS Sources"):
                        for doc in context_docs:
                            src_name = doc.metadata.get("source", "Unknown SDS")
                            raw_page = doc.metadata.get("page", 0)
                            page_num = (raw_page + 1) if isinstance(raw_page, int) else raw_page
                            content_snippet = doc.page_content.strip()

                            st.caption(f"**Source:** `{src_name}` | **Page:** `{page_num}`")
                            st.text(content_snippet)

                            formatted_sources.append({
                                "source": src_name,
                                "page": page_num,
                                "content": content_snippet
                            })

                # Append assistant message with saved sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": formatted_sources
                })

                # Persist conversational memory
                st.session_state.chat_history.extend([
                    HumanMessage(content=query_to_run),
                    AIMessage(content=answer)
                ])

            except Exception as e:
                error_str = str(e)
                if "Connection refused" in error_str or "ConnectError" in error_str:
                    st.error(
                        "❌ Cannot connect to Ollama. Please ensure Ollama is running.\n\n"
                        "Run: `ollama serve`"
                    )
                else:
                    st.error(f"❌ Error generating response: {e}")