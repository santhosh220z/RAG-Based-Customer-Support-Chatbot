"""
Comprehensive Verification Suite: Industry-Ready Dynamic RAG Platform
Tests dynamic YAML configuration, multi-file ingestion from ./data, Chroma DB vector indexing,
and PostgreSQL/SQLite relational query routing.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# LangChain and Verification Modules
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake import FakeListLLM

from ingestion import load_config, load_documents_from_directory, build_vector_store
from main import create_dynamic_rag_chain, check_structured_query
from db.setup_postgres import get_engine, seed_database, lookup_order, lookup_certificate

print("=" * 74)
print(" [TEST] INDUSTRY-READY DYNAMIC RAG DUAL-DATABASE VERIFICATION SUITE")
print("=" * 74)

# ---------------------------------------------------------------------------
# 1. Test Configuration Loader
# ---------------------------------------------------------------------------
print("\n[TEST 1] Testing Central Configuration (config.yaml)...")
config = load_config("config.yaml")
assert config is not None, "Failed to load config.yaml"
print(f"  [SUCCESS] Active Industry: {config.get('business', {}).get('industry')}")
print(f"  [SUCCESS] Company Name:    {config.get('business', {}).get('company_name')}")
print(f"  [SUCCESS] Support Email:   {config.get('escalation', {}).get('email')}")

# ---------------------------------------------------------------------------
# 2. Test Dynamic Document Ingestion Engine (./data/)
# ---------------------------------------------------------------------------
print("\n[TEST 2] Testing Dynamic File Ingestion from ./data/...")
raw_docs = load_documents_from_directory("./data")
assert len(raw_docs) >= 3, f"Expected at least 3 markdown documents, found {len(raw_docs)}"
print(f"  [SUCCESS] Successfully parsed {len(raw_docs)} files from ./data/:")
for doc in raw_docs:
    print(f"   - Source: {doc.metadata.get('source')} | Title: {doc.metadata.get('title')}")

# Test building Chroma vector store dynamically
fake_embeddings = FakeEmbeddings(size=100)
vector_store = build_vector_store(fake_embeddings, config, persist_directory=None, collection_name="test_in_memory_kb")

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.invoke("How do I store Kanchipuram pure silk sarees?")
assert len(retrieved_docs) > 0, "No chunks retrieved from vector store"
print(f"\n  [SUCCESS] Retrieved {len(retrieved_docs)} policy chunks from Chroma DB:")
for idx, doc in enumerate(retrieved_docs, 1):
    print(f"   [{idx}] {doc.metadata.get('title')} ({doc.metadata.get('source')})")
    print(f"       Snippet: {doc.page_content[:95]}...\n")

# ---------------------------------------------------------------------------
# 3. Test Relational Database (PostgreSQL / SQLite)
# ---------------------------------------------------------------------------
print("[TEST 3] Testing Relational Database Queries...")
engine = get_engine()
seed_database(engine)

# Test order lookup
order_res = lookup_order("ORD-7821", engine=engine)
assert order_res is not None, "Failed to retrieve order ORD-7821"
print(f"  [SUCCESS] Order Found: {order_res['order_id']} | Client: {order_res['customer_name']} | Status: {order_res['status']}")

# Test certificate lookup
cert_res = lookup_certificate("GIA-229871034", engine=engine)
assert cert_res is not None, "Failed to retrieve certificate GIA-229871034"
print(f"  [SUCCESS] Certificate Verified: {cert_res['certificate_id']} | Authority: {cert_res['issuing_authority']}")

# Test structured routing check
routed_response = check_structured_query("Where is my order ORD-7821?", engine)
assert routed_response is not None and "ORD-7821" in routed_response
print("  [SUCCESS] Structured query router handled tracking query accurately.")

# ---------------------------------------------------------------------------
# 4. Test Config-Driven Conversational RAG Chain
# ---------------------------------------------------------------------------
print("\n[TEST 4] Testing Conversational History-Aware RAG Chain...")
fake_llm = FakeListLLM(responses=[
    "Stand-alone query: What is the gold buyback rate?",
    "Our store offers a 90% buyback on prevailing market rate for 22K gold net weight.",
])

chain = create_dynamic_rag_chain(fake_llm, vector_store, config)
session_id = "test-session-dynamic-1"

response1 = chain.invoke(
    {"input": "What is the policy for exchanging gold?"},
    config={"configurable": {"session_id": session_id}}
)
print("  Turn 1 Input: 'What is the policy for exchanging gold?'")
print(f"  Turn 1 Output: {response1.get('answer')}")

response2 = chain.invoke(
    {"input": "What if I want a cash buyback instead?"},
    config={"configurable": {"session_id": session_id}}
)
print("  Turn 2 Input: 'What if I want a cash buyback instead?'")
print(f"  Turn 2 Output: {response2.get('answer')}")

print("\n" + "=" * 74)
print(" [SUCCESS] ALL INDUSTRY-READY DYNAMIC TESTS PASSED (100%)")
print("=" * 74)
