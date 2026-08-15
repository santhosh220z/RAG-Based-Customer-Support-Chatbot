# Production-Grade RAG-Based Customer Support Chatbot

An end-to-end, production-grade Customer Support Chatbot powered by modern **LangChain (LCEL / Core)**, **Chroma DB**, and **OpenAI / Google GenAI**.

## 🚀 Features

- **Policy Ingestion & Chunking**: Automatic parsing and chunking of customer support policies (Return & Refund, Shipping & Delivery, Warranty & Repairs, Escalation Guidelines) using `RecursiveCharacterTextSplitter`.
- **Vector Search & Persistence**: Semantic vector search using Chroma DB with metadata filtering.
- **Contextual Multi-Turn Query Reformulation**: History-aware retrieval (`create_history_aware_retriever`) that reformulates conversational follow-up questions in context.
- **Grounded Support Chain**: Strict QA prompt engineering that prevents hallucination, ensures polite & professional responses, and provides contact details for human escalation when context is absent or ambiguous.
- **Multi-Turn Session Memory**: Managed conversational state tracking via `RunnableWithMessageHistory` and session-based history stores.
- **Virtual Environment Auto-Detection**: Built-in verification that automatically checks and ensures the script executes inside `.venv`.
- **Interactive CLI**: Rich terminal chat loop with source citation previews, session IDs, and command shortcuts (`clear`, `exit`).

---

## 📦 Installation & Setup

### 1. Configure Environment Variables
Copy `.env.example` to `.env` and add your API key for OpenAI or Google GenAI:

```bash
# Option 1: OpenAI
OPENAI_API_KEY=sk-...

# Option 2: Google Gemini GenAI
GOOGLE_API_KEY=AIzaSy...
```

### 2. Run the Application

You can launch the chatbot using any of the following methods (the virtual environment will be automatically verified/activated):

#### Option A: Direct Python (Auto-switches to `.venv` if available)
```bash
python main.py
```

#### Option B: PowerShell Runner
```powershell
.\run.ps1
```

#### Option C: Windows Batch
```cmd
run.bat
```

---

## 🛠️ Architecture Overview

```
User Query + Chat History
          │
          ▼
[Contextual Query Reformulation Chain]
  (Reformulates query into standalone question)
          │
          ▼
[Chroma Vector Retriever (k=3)]
  (Retrieves relevant policy chunks)
          │
          ▼
[Grounded Customer Support QA Prompt]
  (System instructions + context + chat history)
          │
          ▼
[LLM (OpenAI gpt-4o-mini / Gemini 1.5 Flash)]
          │
          ▼
[Support Response + Source Citations]
```

---

## 💬 Example Interactions

1. **Initial Inquiry:**
   > *Customer:* "What is your return policy?"
   > *Agent:* "You can return eligible items within 30 days of the delivery date in unused condition with all original packaging..."

2. **Contextual Follow-up:**
   > *Customer:* "Is there a fee for that?"
   > *Agent:* "If the return is due to customer remorse, a flat $5.99 shipping fee is deducted. However, if the item was damaged or defective upon arrival, return shipping is 100% free."

3. **Escalation / Out-of-Scope:**
   > *Customer:* "Can you give me a legal settlement for an accident?"
   > *Agent:* "I do not have specific policy details regarding legal settlements. I would be glad to connect you with our specialized support team. Please reach out to support@company.com or call 1-800-555-0199..."
