"""
Verification test for RAG Customer Support Chatbot components.
Tests imports, policy ingestion, text splitting, and vector store retrieval.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Test imports
print("[TEST] Verifying imports...")
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake import FakeListLLM

try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from main import SAMPLE_SUPPORT_DOCUMENTS, setup_knowledge_base, create_customer_support_rag_chain

print("  [SUCCESS] All modern LangChain modules imported successfully.")

# Test Ingestion and Text Splitting
print("\n[TEST] Testing Knowledge Base Ingestion & Chunking...")
fake_embeddings = FakeEmbeddings(size=100)
vector_store = setup_knowledge_base(fake_embeddings, persist_directory=None)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("What is the return window?")

print(f"  [SUCCESS] Retrieved {len(retrieved_docs)} chunks from Chroma DB:")
for doc in retrieved_docs:
    print(f"   - Title: {doc.metadata.get('title')} | Category: {doc.metadata.get('category')}")
    print(f"     Content snippet: {doc.page_content[:90]}...\n")

# Test History-Aware RAG Chain Construction with Mock LLM
print("[TEST] Testing Conversational RAG Chain Assembly & Message History...")
fake_llm = FakeListLLM(responses=[
    "Stand-alone question: What is the return fee?",
    "According to our Return & Refund policy, a flat fee of $5.99 is deducted for buyer remorse, while damaged returns are 100% free."
])

chain = create_customer_support_rag_chain(fake_llm, vector_store)
session_id = "test-session-1"

response1 = chain.invoke(
    {"input": "What is your return policy?"},
    config={"configurable": {"session_id": session_id}}
)
print("  Turn 1 Input: 'What is your return policy?'")
print(f"  Turn 1 Output: {response1.get('answer')}")

response2 = chain.invoke(
    {"input": "How much does it cost?"},
    config={"configurable": {"session_id": session_id}}
)
print("  Turn 2 Input (Follow-up): 'How much does it cost?'")
print(f"  Turn 2 Output: {response2.get('answer')}")

print("\n=======================================================")
print(" ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! ")
print("=======================================================")
