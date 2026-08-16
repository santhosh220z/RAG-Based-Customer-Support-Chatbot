# Luxury Textile & Fine Jewelry Concierge (Dual-Database RAG System)

A production-grade RAG and Relational Customer Concierge assistant for a **Luxury Textile & Fine Jewelry Store**, powered by **LangChain (LCEL / Core)**, **Chroma DB**, and **PostgreSQL (with SQLite fallback)**.

---

## 🌟 Dual-Database Architecture

```
                                  Client Query
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
     [Unstructured / Policy]                       [Structured / Live Data]
     • Silk, Zari & Pashmina Care Guides           • Order Status & High-Value Tracking
     • 22K/18K Gold Buyback Policies               • Jewelry Certifications (GIA/IGI/BIS)
     • Bespoke Bridal Return Rules                 • Inventory & Custom Tailoring
                │                                             │
                ▼                                             ▼
       [Chroma Vector DB]                           [PostgreSQL Database]
   (Chroma collection & metadata)                 (Relational Tables via SQLAlchemy)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                       [Concierge Response + Cited Sources]
```

---

## 🚀 Key Features

1. **Dual-Database Query Routing**:
   - **Vector Database (Chroma DB)**: Stores and retrieves domain knowledge (BIS 916 Hallmarking, GIA Diamond Grading, Pure Mulberry Silk care, Pashmina wash guide, Lifetime Buyback policies).
   - **Relational Database (PostgreSQL / SQLite)**: Stores structured transactions (Customer VIP tiers, Orders, Jewelry Hallmark Certificates, Order line items).
2. **Contextual Multi-Turn Query Reformulation**: History-aware retrieval (`create_history_aware_retriever`) that reformulates conversational follow-up questions in context.
3. **Luxury Concierge Tone & Anti-Hallucination**: Strict prompt engineering designed for high-value jewelry and luxury textiles with concierge escalation contacts.
4. **Resilient Database Layer**: Automatic fallback to local SQLite (`textile_jewelry_store.db`) if a live PostgreSQL server is not currently reachable.

---

## 📦 Installation & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your API key and database URL:
```env
# Option 1: OpenAI
OPENAI_API_KEY=sk-...

# Option 2: Google Gemini GenAI
# GOOGLE_API_KEY=AIzaSy...

# Optional PostgreSQL Connection (falls back to local SQLite if unreachable)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/textile_jewelry_db
```

### 3. Initialize the Test Database
```bash
python db/setup_postgres.py
```

### 4. Run the Verification Suite
```bash
python verify_rag.py
```

### 5. Launch the Interactive Chatbot
```bash
python main.py
```

---

## 💬 Sample Queries to Test

| Query Type | Example Query | Database Queried |
| :--- | :--- | :--- |
| **Order Status** | *"What is the status of my order ORD-7821?"* | **PostgreSQL** (`orders`, `customers`, `order_items`) |
| **Certificate Verification**| *"Can you verify diamond certificate GIA-229871034?"* | **PostgreSQL** (`jewelry_certifications`, `products`) |
| **Textile Fabric Care** | *"How should I wash my Kanchipuram silk saree?"* | **Chroma DB** (Fabric Care & Preservation) |
| **Gold Buyback & Exchange** | *"What is the policy for 22K gold exchange and buyback?"*| **Chroma DB** (Lifetime Exchange & Buyback) |
| **Bespoke Bridal Tailoring**| *"Can I return a custom-tailored bridal lehenga?"* | **Chroma DB** (Bespoke Bridal & Custom Tailoring) |
