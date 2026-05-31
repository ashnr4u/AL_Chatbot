# src/retriever.py
import os
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD, TOP_K_RETRIEVAL

#using streamlit decorator to cache our embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def retrieve(query, embedding_model, index, chunks, top_k=TOP_K_RETRIEVAL):
    
   
    query_embedding = (
        embedding_model
        .encode([query])  #treated as 1-item batch
        .astype("float32")
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = (
        index.search(query_embedding, top_k)
    )

    retrieved_chunks = []

    for score, idx in zip(scores[0], indices[0]):  
        if idx == -1:
            continue

        if idx >= len(chunks):
            continue

        if score < SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[idx]

        retrieved_chunks.append(
            {
                "text": chunk.page_content,
                "page": chunk.metadata.get("page", "Unknown"),
                "score": float(score)
            }
        )

    return retrieved_chunks