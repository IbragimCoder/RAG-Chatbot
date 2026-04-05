import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text-generation", model="distilgpt2")

# ===== Helper functions =====
def split_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_vector_store(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def search(query, chunks, index, embeddings, k=3):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

st.title("RAG Chatbot (Simple Version)")

text_input = st.text_area("Paste your document here:")

if text_input:
    chunks = split_text(text_input)
    index, embeddings = create_vector_store(chunks)

    question = st.text_input("Ask a question:")

    if question:
        context = search(question, chunks, index, embeddings)
        prompt = f"Context: {' '.join(context)}\nQuestion: {question}\nAnswer:"
        
        response = qa_pipeline(prompt, max_length=150, do_sample=True)[0]["generated_text"]
        
        st.write("### Answer:")
        st.write(response)