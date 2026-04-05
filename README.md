# RAG Chatbot for Documents (LLM Project)

This project implements a simple Retrieval-Augmented Generation (RAG) chatbot using open-source LLMs.

## Description
The system allows users to input a document (text) and ask questions about it. 
The chatbot retrieves relevant parts of the document and generates answers using a language model.

## Features
- Text chunking and preprocessing
- Semantic search using embeddings
- Vector database with FAISS
- LLM-based answer generation
- Interactive UI with Streamlit

## Technologies
- Python
- Hugging Face Transformers
- SentenceTransformers
- FAISS
- Streamlit

## How it works
1. Input text document
2. Split into chunks
3. Convert chunks into embeddings
4. Store in FAISS index
5. Retrieve relevant chunks
6. Generate answer using LLM

## Run
```bash
streamlit run app.py
