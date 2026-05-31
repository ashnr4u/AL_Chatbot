# src/pipeline.py
import os
import re
import tempfile
import json
import faiss
import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_PATH, VECTORDB_PATH, DATA_PATH
from .retriever import load_embedding_model

def clean_text(text):
    #Clean and normalize text
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    return text.strip()

def process_pdf(uploaded_pdf=None, pdf_path=None):
    
    
    # Handle either uploaded file or local file from /data
    if uploaded_pdf is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_pdf.read())
            pdf_path = temp_file.name
    elif pdf_path is not None:
        pdf_path = pdf_path
    else:
        # Look for PDF in /data folder
        data_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        if data_files:
            pdf_path = os.path.join(DATA_PATH, data_files[0])
        else:
            raise FileNotFoundError("No PDF found in /data folder")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)
    
    # Save chunks to /chunks folder
    os.makedirs("chunks", exist_ok=True)
    
    # Save as txt
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.page_content + "\n\n====================\n\n")
    
    # Save as json with metadata
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunks_data.append({
            "index": i,
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", "Unknown"),
            "source": pdf_path
        })
    
    with open("chunks/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2)
    
    # Create embeddings
    model = load_embedding_model()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # cosine similarity setup
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save vector database
    os.makedirs("vectordb", exist_ok=True)
    faiss.write_index(index, VECTORDB_PATH)

    return model, chunks, index