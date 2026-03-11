# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load all PDFs from the data/ folder
print("Loading PDFs...")
documents = []
data_folder = "data"

for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(data_folder, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        # Tag each chunk with its source file name
        for doc in docs:
            doc.metadata["source"] = filename
        documents.extend(docs)
        print(f"  Loaded: {filename} ({len(docs)} pages)")

print(f"\nTotal pages loaded: {len(documents)}")

# 2. Split documents into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# 3. Create embeddings using a free local model
print("\nGenerating embeddings (this takes 1-2 min first time)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 4. Store in FAISS vector database
print("Building vector database...")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Save to disk
vectorstore.save_local("vectorstore")
print("\n✅ Done! Vector database saved to ./vectorstore")