import streamlit as st
import tempfile

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


st.set_page_config(
    page_title="Chemical Safety Assistant",
    page_icon="⚗️",
    layout="centered"
)

st.title("⚗️ Chemical Safety Assistant")

st.markdown(
    "Upload SDS PDFs and ask questions about chemical hazards, storage, PPE, and first aid."
)

st.divider()


# -------------------------
# Upload PDFs
# -------------------------

uploaded_files = st.file_uploader(
    "Upload Safety Data Sheets (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)


# -------------------------
# Process PDFs
# -------------------------

def process_pdfs(files):

    documents = []

    for file in files:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)

        docs = loader.load()

        documents.extend(docs)

    return documents


# -------------------------
# Split documents
# -------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks


# -------------------------
# Create vectorstore
# -------------------------

def create_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# -------------------------
# Create RAG chain
# -------------------------

def create_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOllama(
        model="mistral",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain


# -------------------------
# Build system after upload
# -------------------------

if uploaded_files:

    with st.spinner("Processing documents..."):

        documents = process_pdfs(uploaded_files)

        chunks = split_documents(documents)

        vectorstore = create_vectorstore(chunks)

        chain = create_chain(vectorstore)

        st.session_state.chain = chain

    st.success("Documents processed successfully.")


# -------------------------
# Chat history state
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------
# Display previous messages
# -------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------
# Chat input
# -------------------------

user_question = st.chat_input(
    "Ask a chemical safety question"
)


# -------------------------
# Run query
# -------------------------

if user_question and "chain" in st.session_state:

    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        result = st.session_state.chain(
            {"question": user_question}
        )

        answer = result["answer"]

        st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.divider()

        st.markdown("### Source Documents")

        for doc in result["source_documents"]:

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")

            with st.expander(f"{source} — Page {page}"):

                st.write(doc.page_content)