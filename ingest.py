"""
ingest.py  —  Offline batch ingestion for the Chemical Safety RAG Assistant.

Usage:
    python ingest.py [--data-dir PATH] [--out-dir PATH]

Defaults:
    --data-dir   ./data/       (source PDF directory)
    --out-dir    ./vectorstore/ (FAISS index output)
"""

import argparse
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Shared constants (must match app.py) ─────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-ingest SDS PDFs into a FAISS vector store."
    )
    base = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=base / "data",
        help="Directory containing SDS PDF files (default: ./data/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base / "vectorstore",
        help="Output directory for the FAISS index (default: ./vectorstore/)",
    )
    return parser.parse_args()


def load_pdfs(data_dir: Path) -> list:
    """Load all PDFs from data_dir, tagging each page with its source filename."""
    if not data_dir.exists():
        print(f"❌  Data directory not found: {data_dir}")
        sys.exit(1)

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌  No PDF files found in: {data_dir}")
        sys.exit(1)

    documents = []
    print(f"📂  Loading {len(pdf_files)} PDF file(s) from {data_dir}\n")

    for i, path in enumerate(pdf_files, 1):
        docs = PyPDFLoader(str(path)).load()
        for doc in docs:
            doc.metadata["source"] = path.name
        documents.extend(docs)
        print(f"  [{i}/{len(pdf_files)}] {path.name}  ({len(docs)} pages)")

    print(f"\n✅  Total pages loaded: {len(documents)}")
    return documents


def chunk_documents(documents: list) -> list:
    """Split pages into overlapping chunks for embedding."""
    print(f"\n✂️   Splitting into chunks  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅  Total chunks: {len(chunks)}")
    return chunks


def build_index(chunks: list, out_dir: Path) -> None:
    """Generate embeddings and save the FAISS index to disk."""
    print(f"\n🧠  Loading embedding model: {EMBEDDING_MODEL}…")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("⚙️   Building FAISS index…")
    t0 = time.perf_counter()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_dir))

    print(
        f"✅  Index built in {elapsed:.1f}s  →  saved to {out_dir}\n"
        f"    Vectors stored: {vectorstore.index.ntotal}"
    )


def main() -> None:
    args = parse_args()
    print("=" * 55)
    print("  Chemical Safety RAG Assistant — Batch Ingestion")
    print("=" * 55)

    documents = load_pdfs(args.data_dir)
    chunks = chunk_documents(documents)
    build_index(chunks, args.out_dir)

    print("\n🚀  Done! Run `streamlit run app.py` to start the assistant.")


if __name__ == "__main__":
    main()