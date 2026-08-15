"""
Production-Grade RAG-Based Customer Support Chatbot
Built with LangChain (LCEL / Core), Chroma DB, and OpenAI / Google GenAI.
"""

import os
import sys
import uuid
import subprocess
from typing import Dict, List, Optional
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Virtual Environment Check & Auto-Activation
# ---------------------------------------------------------------------------
def ensure_virtual_environment():
    """Ensure the script runs inside the virtual environment (.venv)."""
    is_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or "VIRTUAL_ENV" in os.environ

    if not is_venv:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python_windows = os.path.join(project_dir, ".venv", "Scripts", "python.exe")
        venv_python_unix = os.path.join(project_dir, ".venv", "bin", "python")
        venv_python = venv_python_windows if os.path.exists(venv_python_windows) else venv_python_unix

        if os.path.exists(venv_python):
            print("[INFO] Virtual environment not active. Re-launching inside '.venv'...")
            result = subprocess.run([venv_python] + sys.argv)
            sys.exit(result.returncode)
        else:
            print("[WARNING] Virtual environment '.venv' not found. Running with global Python interpreter.")


ensure_virtual_environment()

# Load environment variables from .env file
load_dotenv()

# LangChain and Community Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


# ---------------------------------------------------------------------------
# 1. Sample Support Policy Knowledge Base
# ---------------------------------------------------------------------------
SAMPLE_SUPPORT_DOCUMENTS: List[Dict[str, str]] = [
    {
        "title": "Return and Refund Policy",
        "category": "Returns & Refunds",
        "content": (
            "Return Window & Eligibility:\n"
            "Customers may return eligible items within 30 days of the confirmed delivery date. "
            "To qualify for a full refund, items must be unused, in their original condition, and returned "
            "with all original packaging, tags, documentation, and accessories.\n\n"
            "Non-Returnable Items:\n"
            "The following items cannot be returned or refunded: customized or personalized goods, final-sale or "
            "clearance items, opened personal hygiene or health products, gift cards, and downloadable software licenses.\n\n"
            "Return Process & Fees:\n"
            "1. Initiate a return request via the Online Support Portal or contact customer service.\n"
            "2. Download and attach the prepaid return shipping label.\n"
            "3. If the return is due to customer remorse (e.g., changed mind), a flat return shipping fee of $5.99 is deducted from the refund.\n"
            "4. If the item was damaged, defective, or incorrect upon arrival, return shipping is 100% free.\n\n"
            "Refund Processing Time:\n"
            "Once received at our warehouse, items undergo inspection within 3 to 5 business days. "
            "Approved refunds are issued to the original payment method within 5 to 7 business days depending on the financial institution."
        ),
    },
    {
        "title": "Shipping and Delivery Policy",
        "category": "Shipping & Fulfillment",
        "content": (
            "Shipping Methods and Timelines:\n"
            "- Standard Domestic Shipping: 3-5 business days. Cost: $4.99 (Free on all orders over $50.00).\n"
            "- Express Expedited Shipping: 1-2 business days. Cost: $14.99.\n"
            "- International Priority Shipping: 7-14 business days. Rates calculated at checkout based on destination.\n\n"
            "Order Processing Times:\n"
            "Orders placed before 2:00 PM EST Monday through Friday are processed the same business day. "
            "Orders placed after 2:00 PM EST or on weekends/holidays are processed on the next business day.\n\n"
            "International Customs and Duties:\n"
            "International shipments may be subject to customs duties, taxes, and import fees levied by the destination country. "
            "These charges are the sole responsibility of the customer and are not covered in the shipping fee.\n\n"
            "Lost or Delayed Shipments:\n"
            "If a package has not moved or updated for more than 7 business days past the estimated delivery date, "
            "please open a lost shipment inquiry with customer support for an immediate replacement or full refund."
        ),
    },
    {
        "title": "Warranty and Repair Policy",
        "category": "Warranty & Hardware Services",
        "content": (
            "Standard 1-Year Limited Hardware Warranty:\n"
            "All hardware products purchased directly from our store or authorized retailers include a 1-year limited warranty "
            "from the date of purchase covering manufacturing defects in materials and workmanship under ordinary consumer use.\n\n"
            "Warranty Exclusions:\n"
            "The warranty does not cover: accidental damage, drops, liquid immersion or water damage beyond specified IP ratings, "
            "unauthorized repairs or modifications, cosmetic wear and tear, or damage caused by improper electrical voltage/surges.\n\n"
            "Warranty Claim and Repair Procedure:\n"
            "1. Contact technical support with proof of purchase and a detailed description/photo of the defect.\n"
            "2. If approved, customer receives a prepaid shipping box for warranty repair.\n"
            "3. Certified technicians inspect and repair or replace the device within 7 to 10 business days of warehouse arrival.\n"
            "4. Return shipping to the customer is complimentary."
        ),
    },
    {
        "title": "Customer Support and Escalation Guidelines",
        "category": "Customer Care & Escalation",
        "content": (
            "Support Channels and Operating Hours:\n"
            "- Email Support: support@company.com (24/7 ticket submission, response within 24 hours)\n"
            "- Phone Support: 1-800-555-0199 (Mon-Fri 8:00 AM - 8:00 PM EST, Sat 9:00 AM - 5:00 PM EST)\n"
            "- Live Chat: Accessible via web portal during standard business hours.\n\n"
            "Human Agent Escalation Criteria:\n"
            "Our automated assistant must direct customers to a human specialist in the following circumstances:\n"
            "1. If customer query cannot be answered with high confidence using existing published policy.\n"
            "2. If the customer explicitly requests to speak with a human support agent or manager.\n"
            "3. If the case involves unresolved issues pending for over 48 hours.\n"
            "4. If billing/refund disputes exceed $200.00.\n"
            "5. If the customer reports safety incidents, legal claims, or privacy concerns.\n"
            "Escalation contact: support@company.com or phone 1-800-555-0199."
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. Model & Vector Store Initialization
# ---------------------------------------------------------------------------
def initialize_llm_and_embeddings():
    """Select and initialize LLM and Embeddings based on available environment variables."""
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if openai_key:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        print("[INFO] Using OpenAI (gpt-4o-mini / text-embedding-3-small)...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return llm, embeddings

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        print("[INFO] Using Google GenAI (gemini-1.5-flash / text-embedding-004)...")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        return llm, embeddings

    raise ValueError(
        "No LLM API Key detected. Please set either OPENAI_API_KEY or GOOGLE_API_KEY in your environment or .env file."
    )


def setup_knowledge_base(embeddings, persist_directory: Optional[str] = "./chroma_db") -> Chroma:
    """Chunk policy documents and store them in Chroma DB vector store."""
    docs = [
        Document(
            page_content=item["content"],
            metadata={"title": item["title"], "category": item["category"]},
        )
        for item in SAMPLE_SUPPORT_DOCUMENTS
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = splitter.split_documents(docs)

    print(f"[INFO] Ingesting {len(split_docs)} chunks into Chroma DB...")
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


# ---------------------------------------------------------------------------
# 3. History-Aware Retrieval and Grounded RAG Chain
# ---------------------------------------------------------------------------
def create_customer_support_rag_chain(llm, vector_store):
    """Construct a conversational, history-aware RAG pipeline with strict grounding and escalation."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # 1. Contextual query reformulation prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
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

    # 2. Grounded customer support answering prompt
    qa_system_prompt = (
        "You are an expert customer support agent for our company. "
        "Your goal is to assist customers accurately, politely, and concisely.\n\n"
        "Strict Guidelines:\n"
        "1. Base your answer EXCLUSIVELY on the retrieved policy context below. Do not fabricate or assume unstated details.\n"
        "2. If the retrieved context does not contain enough information to answer the question with certainty, "
        "politely inform the customer and offer human support escalation:\n"
        "   - Email: support@company.com (24/7)\n"
        "   - Phone: 1-800-555-0199 (Mon-Fri 8 AM - 8 PM EST)\n"
        "3. Maintain a warm, empathetic, and professional tone at all times.\n"
        "4. Include relevant policy conditions, timeframes, or exceptions when applicable.\n\n"
        "Retrieved Policy Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Combine documents into QA chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create end-to-end retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Session-based multi-turn memory store
    session_store: Dict[str, BaseChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    # Wrap with conversational message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


# ---------------------------------------------------------------------------
# 4. Interactive CLI Interface
# ---------------------------------------------------------------------------
def run_cli_chat(conversational_rag_chain):
    """Run an interactive CLI chat loop with multi-turn memory and source metadata citations."""
    session_id = str(uuid.uuid4())[:8]
    print("\n" + "=" * 70)
    print(" 🤖 CUSTOMER SUPPORT RAG ASSISTANT (Powered by LangChain & Chroma)")
    print("=" * 70)
    print(f"Session ID: {session_id}")
    print("Commands: Type 'exit' or 'quit' to end session | 'clear' to reset history")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("\n👤 Customer: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\n👋 Thank you for contacting customer support. Have a great day!\n")
                break

            if user_input.lower() == "clear":
                session_id = str(uuid.uuid4())[:8]
                print(f"\n[INFO] Conversation history cleared. New Session ID: {session_id}\n")
                continue

            print("\n🔍 Retrieving policy context & generating response...")
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            answer = response.get("answer", "No answer generated.")
            context_docs = response.get("context", [])

            print(f"\n🤖 Support Agent:\n{answer}\n")

            if context_docs:
                print("📚 Sources Cited:")
                seen_sources = set()
                for idx, doc in enumerate(context_docs, 1):
                    title = doc.metadata.get("title", "Policy Document")
                    category = doc.metadata.get("category", "General")
                    source_key = f"{title} ({category})"
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        snippet = doc.page_content.replace("\n", " ")[:140] + "..."
                        print(f"  [{len(seen_sources)}] {title} [{category}] -> \"{snippet}\"")
            print("-" * 70)

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        llm, embeddings = initialize_llm_and_embeddings()
        vector_store = setup_knowledge_base(embeddings)
        support_chain = create_customer_support_rag_chain(llm, vector_store)
        run_cli_chat(support_chain)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
