# app/streamlit_app.py
import json
import os
import pathlib
import pickle

import faiss
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

st.set_page_config(page_title="RAG Q&A Chatbot", page_icon="🤖")
st.title("🤖 RAG Q&A Chatbot")

# Get the root directory
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
INDEX_PATH = ROOT_DIR / "faiss_index.bin"
CHUNKS_PATH = ROOT_DIR / "chunks.pkl"
METADATA_PATH = ROOT_DIR / "metadata.json"

# Load components
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, chunks, metadata

@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

model = load_model()
index, chunks, metadata = load_index()
llm = load_llm()
reranker = load_reranker()

# Verify files loaded
st.sidebar.success(f"✅ Indexed {len(chunks)} chunks")
st.sidebar.info(f"📊 Index size: {os.path.getsize(INDEX_PATH)/1024**2:.1f} MB")

# Functions
def retrieve(query: str, top_k: int = 10):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "chunk": chunks[idx],
            "initial_score": float(score),
            "metadata": metadata[idx]
        })
    return results

def retrieve_with_rerank(query: str, top_k: int = 10, rerank_top_k: int = 5):
    candidates = retrieve(query, top_k=top_k)
    if len(candidates) == 0:
        return []
    
    pairs = [[query, c["chunk"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    
    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)
    
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:rerank_top_k]

def generate_answer(question: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful Q&A assistant. Answer ONLY using the context below. "
        "If the context does not contain the answer, say 'I don't know based on the provided context.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context_text})
    return response.content if hasattr(response, "content") else str(response)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything from your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved = retrieve_with_rerank(prompt, top_k=10, rerank_top_k=5)
            contexts = [r["chunk"] for r in retrieved]
            answer = generate_answer(prompt, contexts)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        with st.expander("🔍 Retrieved Context & Sources"):
            for i, r in enumerate(retrieved, 1):
                st.markdown(f"**Result {i} (score: {r['rerank_score']:.3f})**")
                st.markdown(f"**Title:** {r['metadata'].get('title', 'N/A')}")
                st.markdown(f"**Subject:** {r['metadata'].get('subject', 'N/A')}")
                st.text(r['chunk'][:400] + ("..." if len(r['chunk']) > 400 else ""))