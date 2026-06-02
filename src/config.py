# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 10
SIMILARITY_THRESHOLD = 0.50

# Paths
DATA_PATH = "data"
CHUNKS_PATH = "chunks/chunks.txt"
VECTORDB_PATH = "vectordb/index.faiss"
