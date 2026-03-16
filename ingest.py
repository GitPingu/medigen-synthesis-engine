"""
Ingest MediGen synthetic documents into ChromaDB vector store.
Uses ChromaDB's built-in sentence-transformer embeddings (runs locally, no API needed).
"""
import os
import chromadb
from chromadb.config import Settings

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "demo_corpus")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def chunk_document(text, filename, max_chars=1500):
    """Split a document into chunks, preserving section boundaries."""
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        # Start new chunk at section headers or when size exceeded
        is_header = (
            line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7."))
            and len(line.strip()) < 100
        )
        if is_header and current_len > 300:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(line)
        current_len += len(line)

        if current_len > max_chars:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def extract_metadata(text, filename):
    """Extract basic metadata from document content."""
    meta = {"filename": filename, "source": "unknown", "department": "unknown"}

    # Extract department
    dept_map = {
        "research": "Research",
        "clinical": "Clinical Development",
        "regulatory": "Regulatory Affairs",
        "legal": "Legal",
        "manufacturing": "Manufacturing & CMC",
        "quality": "Quality",
        "medaffairs": "Medical Affairs",
        "commercial": "Commercial",
        "it": "IT",
        "misc": "Corporate",
    }
    for key, val in dept_map.items():
        if f"_{key}_" in filename:
            meta["department"] = val
            break

    # Extract system
    for system in [
        "Benchling",
        "SharePoint",
        "S: drive",
        "S_drive",
        "Veeva Vault",
        "CPA Global",
        "Ironclad",
        "MasterControl",
    ]:
        if system in text:
            meta["source"] = system.replace("S_drive", "S: drive")
            break

    # Extract product/program
    for prog in [
        "VELORIN",
        "CARALYN",
        "MG-401",
        "MG-309",
        "MG-217",
        "MG-Link",
    ]:
        if prog in text:
            meta["program"] = prog
            break

    return meta


def ingest():
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists
    try:
        client.delete_collection("medigen_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="medigen_docs",
        metadata={"description": "MediGen Corp internal document corpus"},
    )

    doc_files = sorted(
        [f for f in os.listdir(CORPUS_DIR) if f.endswith(".md")]
    )
    print(f"Found {len(doc_files)} documents to ingest.")

    all_ids = []
    all_texts = []
    all_metas = []

    for i, filename in enumerate(doc_files):
        filepath = os.path.join(CORPUS_DIR, filename)
        with open(filepath, "r") as f:
            text = f.read()

        metadata = extract_metadata(text, filename)
        chunks = chunk_document(text, filename)

        for j, chunk in enumerate(chunks):
            chunk_id = f"{filename}::chunk_{j}"
            chunk_meta = {**metadata, "chunk_index": j, "total_chunks": len(chunks)}
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metas.append(chunk_meta)

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(doc_files)} documents...")

    # Add in batches (ChromaDB limit)
    batch_size = 100
    for start in range(0, len(all_ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            metadatas=all_metas[start:end],
        )

    print(f"Ingested {len(all_ids)} chunks from {len(doc_files)} documents.")
    print(f"ChromaDB persisted to: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
