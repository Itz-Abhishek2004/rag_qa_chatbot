# 🤖 RAG Q&A Chatbot

**Retrieval-Augmented Generation (RAG)** chatbot for intelligent question-answering over large document collections.

## 🎯 Project Overview

This project implements a production-ready RAG pipeline that retrieves relevant context from 21,417 news articles (43,516 chunks) and generates accurate, grounded answers using LLMs.

### Key Features

- ✅ **Semantic Search** — FAISS vector index with 43,516 chunks
- ✅ **Reranking** — Cross-encoder for improved retrieval accuracy
- ✅ **Low Latency** — Average retrieval time: 1.1s
- ✅ **High Accuracy** — Average rerank score: 1.274
- ✅ **Explainable** — Source citations with relevance scores
- ✅ **Scalable** — 126 MB index, easily deployable

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Dataset** | 21,417 rows, 82.65 MB (News articles) |
| **Total Chunks** | 43,516 (2.03 chunks/row) |
| **Embedding Model** | `all-MiniLM-L6-v2` (384 dimensions) |
| **Vector Index** | FAISS `IndexFlatIP` (63.74 MB) |
| **Reranking Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Average Retrieval Score** | 1.274 |
| **Average Retrieval Time** | 1.097s |
| **LLM** | Groq + Llama 3.1 8B |
| **Total Storage** | 126.49 MB |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **NLP/ML:** `sentence-transformers`, `transformers`, `langchain`
- **Vector DB:** `faiss-cpu`
- **LLM:** Groq (Llama 3.1 8B) via `langchain-groq`
- **App Framework:** `streamlit`
- **Evaluation:** `evaluate`, `rouge-score`, `bertscore`

---

## 📁 Project Structure

```text
rag_qa_chatbot/
├─ data/
│  └─ true.csv                    # 21,417 news articles
├─ notebooks/
│  └─ 01_rag_pipeline.ipynb       # Complete RAG pipeline
├─ app/
│  └─ streamlit_app.py            # Web application
├─ faiss_index.bin                 # 63.74 MB
├─ chunks.pkl                      # 54.20 MB
├─ metadata.json                   # 8.55 MB
├─ evaluation_results.csv          # Test query results
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/rag_qa_chatbot.git
cd rag_qa_chatbot
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

Create `.env` file:

```bash
echo "GROQ_API_KEY=your_groq_key_here" > .env
```

Get free API key: https://console.groq.com

### 4. Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open browser: http://localhost:8501

---

## 📥 Download Pre-built Index (Optional)

Large files are not included due to GitHub's 100MB limit.

Download from Google Drive:
- [faiss_index.bin, chunks.pkl, metadata.json](https://drive.google.com/drive/folders/1MF0aKAxahrU-VgU1HyLiHZNfWdoDLK9z?usp=sharing)

Place them in the project root directory.

## 📖 How It Works

### 1. Data Pipeline

- **Ingestion:** Load CSV (21,417 rows, 82.65 MB)
- **Cleaning:** Remove URLs, special characters, normalize whitespace
- **Chunking:** 300-token chunks with 50-token overlap → 43,516 chunks

### 2. Embeddings & Indexing

- **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions)
- **Vector Index:** FAISS `IndexFlatIP` (cosine similarity)
- **Index Size:** 63.74 MB

### 3. Retrieval

- **Semantic Search:** Top-10 chunks via cosine similarity
- **Reranking:** Cross-encoder reranks to top-5
- **Latency:** ~1.1s average

### 4. Generation

- **LLM:** Groq + Llama 3.1 8B
- **Prompt:** Context-augmented with retrieved chunks
- **Output:** Grounded answer with source citations

---

## 🧪 Evaluation

### Test Queries (10 samples)

| Query | Top Score | Retrieval Time |
|-------|-----------|----------------|
| What is Republican budget policy? | 8.314 | 1.2s |
| What does Trump say about Amazon? | 2.156 | 1.1s |
| What are education spending priorities? | 1.842 | 1.0s |
| What is the news about tax cuts? | 1.654 | 1.1s |
| What did Republicans say about fiscal conservative? | 1.523 | 1.0s |

**Average Retrieval Score:** 1.274  
**Average Retrieval Time:** 1.097s

---

## 🎯 Business Use Cases

- **Customer Support:** Automated Q&A from product manuals
- **Legal Firms:** Querying contracts or case files
- **Education:** Answering student questions from textbooks
- **Enterprises:** Internal knowledge base search
- **Research:** Extracting insights from academic papers

---

## 📈 Optimization Techniques

1. **Chunking Strategy:** 300 tokens with 50-token overlap
2. **Embedding Model:** `all-MiniLM-L6-v2` for speed-accuracy balance
3. **Reranking:** Cross-encoder for improved relevance
4. **FAISS Index:** In-memory `IndexFlatIP` for fast retrieval
5. **Batch Processing:** `batch_size=128` for efficient embedding generation

---

## 🚀 Future Improvements

- [ ] **Hybrid Search:** Combine BM25 (keyword) + Vector (semantic)
- [ ] **Query Rewriting:** HyDE technique for better retrieval
- [ ] **Parent-Child Chunking:** Hierarchical retrieval
- [ ] **Caching:** Redis for frequent queries
- [ ] **Deployment:** Docker + AWS EC2 / GCP Cloud Run
- [ ] **Monitoring:** LangSmith for tracing and analytics

---

## 👤 Author

**Abhishek Sahani**  
[LinkedIn](https://www.linkedin.com/in/abhishek-sahani-b1bb33292/) 
---

## 🙏 Acknowledgments

- Dataset: Natural Questions (NQ) style news articles
- Models: Sentence Transformers, FAISS, Groq Llama 3
- Frameworks: LangChain, Streamlit
