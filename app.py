# app.py
import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


st.set_page_config(page_title="Chemical Safety Assistant", page_icon="⚗️", layout="centered")
st.title("⚗️ Chemical Safety Assistant")
st.markdown("Ask questions about chemical hazards, protective equipment, storage, and first aid.")
st.divider()

@st.cache_resource
def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt_template = """You are a chemical safety expert assistant. 
Use only the context below to answer the question. 
If the answer is not in the context, say "I don't have enough information in the loaded documents to answer this."

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOllama(
    model="mistral",
    temperature=0
)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

with st.spinner("Loading knowledge base..."):
    chain = load_rag_chain()

st.markdown("**💡 Example questions:**")
examples = [
    "What are the hazards of acetone?",
    "What protective equipment is required for benzene?",
    "What is the flash point of ethanol?",
    "What should I do if methanol is ingested?",
    "How should acetone be stored?"
]
cols = st.columns(2)
for i, example in enumerate(examples):
    if cols[i % 2].button(example, key=f"ex_{i}"):
        st.session_state["query"] = example

query = st.text_input(
    "Ask a question:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What are the fire hazards of methanol?"
)

if st.button("🔍 Search", type="primary") and query:
    with st.spinner("Searching documents and generating answer..."):
        result = chain.invoke({"query": query})

    st.markdown("### 📋 Answer")
    st.markdown(result["result"])

    st.divider()
    st.markdown("### 📄 Source Documents Used")
    seen = set()
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            with st.expander(f"📎 {source} — Page {page}"):
                st.markdown(doc.page_content)