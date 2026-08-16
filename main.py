"""
Industry-Ready RAG Customer Concierge System
Dynamically configured via config.yaml, loads docs via ingestion.py, with Dual Chroma DB + PostgreSQL support.
"""

import os
import sys
import re
import uuid
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables from .env file
load_dotenv()

# LangChain Core Imports
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Import Ingestion and Relational Database Modules
from ingestion import load_config, build_vector_store
from db.setup_postgres import lookup_order, lookup_certificate, get_engine, seed_database


# ---------------------------------------------------------------------------
# 1. LLM & Embeddings Initialization (Hugging Face / OpenAI / Gemini)
# ---------------------------------------------------------------------------
def initialize_llm_and_embeddings(config: Optional[Dict[str, Any]] = None):
    """Initialize LLM and Embeddings from Hugging Face, OpenAI, or Google GenAI."""
    if config is None:
        config = load_config()

    model_cfg = config.get("model", {})
    provider = model_cfg.get("provider", "huggingface").lower()
    temperature = model_cfg.get("temperature", 0.1)

    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    # Option A: Explicit OpenAI requested
    if provider == "openai" and openai_key and not openai_key.startswith("your_"):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        model_name = model_cfg.get("openai_model", "gpt-4o-mini")
        embed_name = model_cfg.get("openai_embeddings", "text-embedding-3-small")
        print(f"[INFO] Using OpenAI ({model_name} / {embed_name})...")
        llm = ChatOpenAI(model=model_name, temperature=temperature)
        embeddings = OpenAIEmbeddings(model=embed_name)
        return llm, embeddings

    # Option B: Explicit Google Gemini requested
    if provider == "gemini" and google_key and not google_key.startswith("your_"):
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        model_name = model_cfg.get("gemini_model", "gemini-1.5-flash")
        embed_name = model_cfg.get("gemini_embeddings", "models/text-embedding-004")
        print(f"[INFO] Using Google GenAI ({model_name} / {embed_name})...")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        embeddings = GoogleGenerativeAIEmbeddings(model=embed_name)
        return llm, embeddings

    # Option C: Hugging Face Local Pipeline (100% Free, Zero API Keys Required)
    print("\n[INFO] Initializing local Hugging Face model & embeddings (Zero API Keys required)...")
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    hf_model_id = model_cfg.get("hf_model_id", "Qwen/Qwen2.5-1.5B-Instruct")
    hf_embed_id = model_cfg.get("hf_embeddings_id", "sentence-transformers/all-MiniLM-L6-v2")
    max_tokens = model_cfg.get("max_new_tokens", 512)

    print(f"  [1/2] Loading Hugging Face Embeddings ({hf_embed_id})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=hf_embed_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"  [2/2] Loading Hugging Face LLM Pipeline ({hf_model_id})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optimize CPU multi-threading
    if device == "cpu":
        cpu_cores = os.cpu_count() or 4
        torch.set_num_threads(cpu_cores)
        print(f"  [INFO] PyTorch multi-threading enabled with {cpu_cores} CPU threads.")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    # Dynamic INT8 CPU Quantization for ~2x speedup and lower memory bandwidth
    if device == "cpu":
        try:
            print("  [INFO] Applying dynamic INT8 quantization for accelerated CPU inference...")
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as q_err:
            print(f"  [NOTICE] Quantization skipped: {q_err}")

    # Fast greedy decoding (do_sample=False) + token limit for instant responses
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=None if temperature == 0 else temperature,
        do_sample=False,  # Greedy decoding is significantly faster on CPU
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        truncation=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"  [SUCCESS] Accelerated Hugging Face pipeline ready on device: {device.upper()}\n")
    return llm, embeddings


# ---------------------------------------------------------------------------
# 2. Dynamic History-Aware RAG Chain Construction
# ---------------------------------------------------------------------------
def create_dynamic_rag_chain(llm, vector_store, config: Dict[str, Any]):
    """Construct a conversational, history-aware RAG pipeline driven by config.yaml."""
    vector_cfg = config.get("vector_db", {})
    business_cfg = config.get("business", {})
    escalation_cfg = config.get("escalation", {})

    top_k = vector_cfg.get("top_k_retrieval", 3)
    company_name = business_cfg.get("company_name", "Our Company")
    assistant_title = business_cfg.get("assistant_title", "Customer Concierge")
    tone = business_cfg.get("tone", "polite, helpful, and professional")
    support_email = escalation_cfg.get("email", "support@company.com")
    support_phone = escalation_cfg.get("phone", "1-800-555-0199")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # 1. Contextual query reformulation prompt
    contextualize_q_system_prompt = (
        f"Given a chat history and the latest user question regarding {company_name}, "
        "formulate a standalone question that can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # 2. Concise and Grounded QA prompt
    qa_system_prompt = (
        f"You are the {assistant_title} for {company_name}.\n"
        f"Tone: {tone}.\n"
        "Guidelines:\n"
        "- Answer the client's question accurately using ONLY the store policy context below.\n"
        "- Be concise, polite, and direct (2-4 clear sentences).\n"
        f"- If the answer is not in the context, invite them to contact our concierge at {support_email} or {support_phone}.\n\n"
        "Store Policy Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    session_store: Dict[str, BaseChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


# ---------------------------------------------------------------------------
# 3. Structured Database Routing (Orders & Certifications)
# ---------------------------------------------------------------------------
def check_structured_query(user_query: str, db_engine) -> Optional[str]:
    """Check if query is asking for a specific order ID or certificate number in PostgreSQL/SQLite."""
    # Check for Order ID pattern (e.g., ORD-7821 or tracking ARM-SEC-99812)
    order_match = re.search(r"\b(ORD-\d+|ARM-SEC-\d+|FEDEX-EXP-\d+)\b", user_query, re.IGNORECASE)
    if order_match:
        order_key = order_match.group(1).upper()
        order_info = lookup_order(order_key, engine=db_engine)
        if order_info:
            return (
                f"📦 **Order Status for {order_info['order_id']}**:\n"
                f"- **Client:** {order_info['customer_name']} (VIP Tier: {order_info['vip_tier']})\n"
                f"- **Order Date:** {order_info['order_date']}\n"
                f"- **Current Status:** `{order_info['status']}`\n"
                f"- **Tracking Number:** {order_info['tracking_number']}\n"
                f"- **Total Amount:** ${order_info['total_amount']:,.2f}\n"
                f"- **Items Ordered:**\n{order_info['items']}"
            )

    # Check for Certificate pattern (e.g., GIA-229871034 or BIS-916-ND-4491)
    cert_match = re.search(r"\b(GIA-\d+|BIS-916-[A-Z0-9-]+)\b", user_query, re.IGNORECASE)
    if cert_match:
        cert_key = cert_match.group(1).upper()
        cert_info = lookup_certificate(cert_key, engine=db_engine)
        if cert_info:
            specs = []
            if cert_info["carat_weight"]:
                specs.append(f"Carat Weight: {cert_info['carat_weight']} ct")
            if cert_info["clarity_grade"]:
                specs.append(f"Clarity: {cert_info['clarity_grade']}")
            if cert_info["cut_grade"]:
                specs.append(f"Cut: {cert_info['cut_grade']}")
            if cert_info["gold_hallmark_id"]:
                specs.append(f"Gold Hallmark ID: {cert_info['gold_hallmark_id']}")

            return (
                f"💎 **Authenticity Certificate Verified ({cert_info['certificate_id']})**:\n"
                f"- **Issuing Authority:** {cert_info['issuing_authority']}\n"
                f"- **Associated Item:** {cert_info['product_name']}\n"
                f"- **Verification Date:** {cert_info['verified_date']}\n"
                f"- **Certified Specifications:** {', '.join(specs)}"
            )

    return None


# ---------------------------------------------------------------------------
# 4. Interactive CLI Interface
# ---------------------------------------------------------------------------
def run_cli_chat(conversational_rag_chain, db_engine, config: Dict[str, Any]):
    """Run an interactive CLI chat loop with dual-database querying."""
    business_cfg = config.get("business", {})
    welcome_banner = business_cfg.get(
        "welcome_banner", "🤖 CUSTOMER SUPPORT CONCIERGE (Dual-Database RAG System)"
    )

    session_id = str(uuid.uuid4())[:8]
    print("\n" + "=" * 76)
    print(f" {welcome_banner}")
    print("=" * 76)
    print(f"Session ID: {session_id} | Industry: {business_cfg.get('industry', 'General')}")
    print("Commands: Type 'exit' or 'quit' to end session | 'clear' to reset history")
    print("=" * 76 + "\n")

    while True:
        try:
            user_input = input("\n👤 Client: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\n👋 Thank you for contacting our concierge. Have a wonderful day!\n")
                break

            if user_input.lower() == "clear":
                session_id = str(uuid.uuid4())[:8]
                print(f"\n[INFO] Conversation history reset. New Session ID: {session_id}\n")
                continue

            # 1. Check if structured data lookup is present in PostgreSQL/SQLite
            direct_db_result = check_structured_query(user_input, db_engine)
            if direct_db_result:
                print(f"\n🤖 Concierge:\n{direct_db_result}\n")
                print("📚 Source: PostgreSQL / Relational Database (Verified Store Records)")
                print("-" * 76)
                continue

            # 2. Semantic Search via Chroma Vector DB & LLM
            print("\n🔍 Searching knowledge base & generating response...")
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            raw_answer = response.get("answer", "No answer generated.")
            
            # Clean up residual artifacts and formatting tags
            clean_answer = re.sub(r"<\|im_start\|>assistant\s*", "", raw_answer)
            clean_answer = re.sub(r"<\|im_end\|>.*", "", clean_answer, flags=re.DOTALL)
            clean_answer = re.sub(r"^Assistant:\s*", "", clean_answer, flags=re.IGNORECASE).strip()

            context_docs = response.get("context", [])

            print(f"\n🤖 Concierge:\n{clean_answer}\n")

            if context_docs:
                print("📚 Sources Cited (Chroma DB):")
                seen_sources = set()
                for doc in context_docs:
                    title = doc.metadata.get("title") or doc.metadata.get("source", "Document")
                    category = doc.metadata.get("category", "General")
                    source_key = f"{title} ({category})"
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        snippet = doc.page_content.replace("\n", " ")[:140] + "..."
                        print(f"  [{len(seen_sources)}] {title} [{category}] -> \"{snippet}\"")
            print("-" * 76)

        except KeyboardInterrupt:
            print("\n\n👋 Concierge session ended. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config("config.yaml")

        # Initialize PostgreSQL / SQLite Relational Database
        engine = get_engine()
        seed_database(engine)

        # Initialize LLM & Dynamic Chroma Vector DB
        llm, embeddings = initialize_llm_and_embeddings(config)
        vector_store = build_vector_store(embeddings, config)
        support_chain = create_dynamic_rag_chain(llm, vector_store, config)

        run_cli_chat(support_chain, engine, config)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
