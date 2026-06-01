# PDF RAG Chatbot with Streaming Responses


## Demo

Video Demo: <https://drive.google.com/file/d/1huUK3IX7Zx3be7BA8DW9S5R-bfIjnXES/view?usp=sharing>  
Screenshots: <https://drive.google.com/file/d/15lIUxAv5ZxpFaDpCAzQ-wphhLOXq7MtE/view?usp=sharing>

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot capable of answering questions from uploaded PDF documents. The system combines semantic search using vector embeddings with a Large Language Model (LLM) to provide accurate, context-aware responses.

The chatbot is built using Streamlit, FAISS, Sentence Transformers, and Groq-hosted Llama 3.3, with support for real-time streaming responses and source chunk visualization.

## Features

- PDF document upload and processing
- Text cleaning and chunking
- Semantic embeddings using BGE Small
- FAISS vector database for retrieval
- Retrieval-Augmented Generation (RAG) pipeline
- Real-time streaming responses
- Source chunk display with relevance scores
- Chat history support
- Simple and interactive Streamlit interface

## System Architecture

## System Architecture

PDF Upload → Text Extraction → Cleaning → Chunking → Embedding Generation → FAISS Indexing → Query Embedding →       Semantic Retrieval → Prompt Construction → LLM Generation → Streaming Response

## Technologies Used

Embedding Model: BAAI/bge-small-en-v1.5
- Selected for its strong semantic retrieval performance
- Lightweight and efficient for CPU/GPU deployment
- Performs well on sentence-level similarity tasks

LLM: llama-3.3-70b-versatile (Groq API)
- High-quality instruction-following capability
- Fast inference through Groq’s optimized API
- Strong reasoning and context adherence for RAG tasks

Vector Database: FAISS
- Fast similarity search
- Local deployment
- Efficient retrieval for RAG applications

## Installation

Clone Repository
Step 1: Clone the Repository
    bash
    git clone <your-repository-url>
    cd <repository-name>

Step 2: Create Virtual Environment (Recommended)
```
    python -m venv venv
    Activate it:
    Windows:venv\Scripts\activate
    Mac / Linux:source venv/bin/activate
```
Step 3: Install Dependencies
```
    pip install -r requirements.txt
```
Step 4: Configure Environment Variables
```
    Create a .env file in the root directory:
    GROQ_API_KEY=your_api_key_here
```
## Running the Application

Launch the Streamlit application:
```
bash
python -m streamlit run app.py
```
Then:

1. Upload a PDF document
2. Wait for indexing to complete
3. Ask questions related to the document
4. View retrieved source chunks used to generate answers

## RAG Pipeline

Document Processing

1. PDF extraction using PyPDFLoader
2. Text cleaning using regular expressions
3. Document chunking using RecursiveCharacterTextSplitter
4. Embedding generation using BGE Small
5. Storage in FAISS vector index

Retrieval

1. User query is converted into an embedding
2. FAISS performs similarity search
3. Top relevant chunks are retrieved

Generation

1. Retrieved chunks are injected into the prompt
2. Llama 3.3 generates responses using only retrieved context
3. Responses are streamed in real time

## Prompt Strategy

The model is strictly grounded in retrieved context and is instructed to avoid hallucination.
- Answer only using provided context
- Do not use external knowledge
- Do not make assumptions
- If information is unavailable, return: NOT FOUND IN DOCUMENT


## Project Structure

```
AL_chatbot/
│
├── chunks/              # Processed and stored text chunks
├── data/                # Raw uploaded PDF documents
├── notebooks/           # Experiments and preprocessing notebooks
├── vectordb/            # FAISS vector database storage
│
├── src/                 # Core RAG pipeline modules
│   ├── __init__.py
│   ├── config.py        # Configuration settings (paths, models, constants)
│   ├── retriever.py     # FAISS-based semantic retrieval logic
│   ├── generator.py     # LLM prompt construction + response generation
│   └── pipeline.py      # End-to-end RAG pipeline orchestration
│
├── app.py               # Streamlit chatbot application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .env                 # Environment variables (API keys)
```
## Limitations

- Processes one PDF at a time
- Retrieval quality depends on chunking strategy
- Large PDFs may require longer indexing times
- Responses are limited to retrieved context


## Author

Ashutosh Narayan