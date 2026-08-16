"""
Dynamic Document Ingestion Engine for Industry-Ready RAG Chatbot
Scans and parses .md, .txt, .pdf, and .json files from the configured data directory.
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with default fallback."""
    if not os.path.exists(config_path):
        return {
            "business": {
                "company_name": "Customer Support Concierge",
                "assistant_title": "Support Concierge",
                "tone": "polite, helpful, and professional",
                "welcome_banner": "🤖 CUSTOMER SUPPORT CONCIERGE",
            },
            "escalation": {
                "email": "support@company.com",
                "phone": "1-800-555-0199",
            },
            "ingestion": {
                "data_dir": "./data",
                "chunk_size": 450,
                "chunk_overlap": 60,
            },
            "vector_db": {
                "persist_directory": "./chroma_db",
                "collection_name": "store_knowledge_base",
                "top_k_retrieval": 3,
            },
        }

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_title_from_text(text: str, filename: str) -> str:
    """Extract a human-friendly title from markdown headers or filename."""
    lines = text.strip().split("\n")
    for line in lines[:5]:
        if line.startswith("# "):
            return line.replace("# ", "").strip()
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def load_documents_from_directory(data_dir: str) -> List[Document]:
    """Load all supported documents (.md, .txt, .pdf, .json) from data directory."""
    documents: List[Document] = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"[WARNING] Data directory '{data_dir}' does not exist.")
        return documents

    for file_path in data_path.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        rel_path = file_path.name

        try:
            # 1. Markdown & Text files
            if ext in (".md", ".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        title = extract_title_from_text(content, rel_path)
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": rel_path,
                                "title": title,
                                "category": file_path.parent.name if file_path.parent != data_path else "General Policy",
                            },
                        )
                        documents.append(doc)

            # 2. PDF Documents
            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(file_path))
                    pdf_text = []
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            pdf_text.append(text)

                    full_pdf_text = "\n\n".join(pdf_text).strip()
                    if full_pdf_text:
                        doc = Document(
                            page_content=full_pdf_text,
                            metadata={
                                "source": rel_path,
                                "title": extract_title_from_text(full_pdf_text, rel_path),
                                "category": "PDF Documentation",
                                "page_count": len(reader.pages),
                            },
                        )
                        documents.append(doc)
                except Exception as pdf_err:
                    print(f"[ERROR] Failed to parse PDF {rel_path}: {pdf_err}")

            # 3. JSON Knowledge Files
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for idx, item in enumerate(data):
                            content = item.get("content") or json.dumps(item, indent=2)
                            title = item.get("title") or f"{rel_path} Entry {idx+1}"
                            category = item.get("category") or "Knowledge Base"
                            documents.append(
                                Document(
                                    page_content=content,
                                    metadata={"source": rel_path, "title": title, "category": category},
                                )
                            )
                    elif isinstance(data, dict):
                        content = data.get("content") or json.dumps(data, indent=2)
                        title = data.get("title") or Path(rel_path).stem.title()
                        category = data.get("category") or "Knowledge Base"
                        documents.append(
                            Document(
                                page_content=content,
                                metadata={"source": rel_path, "title": title, "category": category},
                            )
                        )

        except Exception as e:
            print(f"[ERROR] Error loading file '{rel_path}': {e}")

    return documents


def build_vector_store(
    embeddings,
    config: Optional[Dict[str, Any]] = None,
    persist_directory: Optional[str] = "./chroma_db",
    collection_name: Optional[str] = None,
) -> Chroma:
    """Load documents, chunk them, and store inside Chroma DB."""
    if config is None:
        config = load_config()

    ingestion_cfg = config.get("ingestion", {})
    vector_cfg = config.get("vector_db", {})

    data_dir = ingestion_cfg.get("data_dir", "./data")
    chunk_size = ingestion_cfg.get("chunk_size", 450)
    chunk_overlap = ingestion_cfg.get("chunk_overlap", 60)
    separators = ingestion_cfg.get("separators", ["\n\n", "\n", ". ", " ", ""])

    persist_dir = persist_directory
    target_collection = collection_name or vector_cfg.get("collection_name", "store_knowledge_base")

    # Load raw documents
    raw_docs = load_documents_from_directory(data_dir)

    if not raw_docs:
        print("[WARNING] No documents found in data directory. Adding fallback knowledge...")
        raw_docs = [
            Document(
                page_content="Welcome to our customer concierge. Please contact support@company.com for inquiries.",
                metadata={"source": "default", "title": "General Inquiries", "category": "General"},
            )
        ]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    split_docs = splitter.split_documents(raw_docs)

    print(f"[INFO] Ingesting {len(raw_docs)} files ({len(split_docs)} chunks) into Chroma DB ({persist_dir})...")

    try:
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=target_collection,
            persist_directory=persist_dir,
        )
    except Exception as e:
        if "dimension" in str(e).lower():
            print(f"[WARNING] Detected embedding dimension mismatch ({e}). Resetting vector store collection...")
            import shutil
            if persist_dir and os.path.exists(persist_dir):
                shutil.rmtree(persist_dir, ignore_errors=True)
            vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                collection_name=target_collection,
                persist_directory=persist_dir,
            )
        else:
            raise e

    return vector_store
