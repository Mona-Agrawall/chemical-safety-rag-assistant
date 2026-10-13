# ingest.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

# 1. Load all PDFs from the data/ folder
print("Loading PDFs...")
documents = []

if not DATA_PATH.exists():
    print(f"❌ Data folder not found: {DATA_PATH}")
    print("   Create a 'data' folder and add SDS PDF files to it.")
    exit(1)

pdf_files = list(DATA_PATH.glob("*.pdf"))

if not pdf_files:
    print(f"❌ No PDF files found in {DATA_PATH}")
    exit(1)

for filepath in pdf_files:
    loader = PyPDFLoader(str(filepath))
    docs = loader.load()
    # Tag each chunk with its source file name
    for doc in docs:
        doc.metadata["source"] = filepath.name
    documents.extend(docs)
    print(f"  Loaded: {filepath.name} ({len(docs)} pages)")

print(f"\nTotal pages loaded: {len(documents)}")

# 2. Split documents into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
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
VECTORSTORE_PATH.mkdir(exist_ok=True)
vectorstore.save_local(str(VECTORSTORE_PATH))
print(f"\n✅ Done! Vector database saved to {VECTORSTORE_PATH}")