# PDF RAG Chatbot with Streaming Responses

## Demo

**Video Demo:** https://drive.google.com/file/d/1huUK3IX7Zx3be7BA8DW9S5R-bfIjnXES/view?usp=sharing

**Screenshots:** https://drive.google.com/file/d/15lIUxAv5ZxpFaDpCAzQ-wphhLOXq7MtE/view?usp=sharing

---

# Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot capable of answering questions from uploaded PDF documents. The system combines semantic search using vector embeddings with a Large Language Model (LLM) to provide accurate, context-aware responses.

The chatbot is built using **Streamlit**, **FAISS**, **Sentence Transformers**, and **Groq-hosted Llama 3.3**, with support for:

* Real-time streaming responses
* Source chunk visualization
* Automated RAG evaluation using an LLM-as-a-Judge pipeline

---

# Features

* PDF document upload and processing
* Text cleaning and chunking
* Semantic embeddings using BGE Small
* FAISS vector database for retrieval
* Retrieval-Augmented Generation (RAG)
* Real-time streaming responses
* Source chunk display with similarity scores
* Chat history support
* Automated RAG evaluation pipeline
* LLM-based answer quality scoring

---

# System Architecture

```
PDF Upload
      │
      ▼
Text Extraction
      │
      ▼
Cleaning
      │
      ▼
Chunking
      │
      ▼
Embedding Generation
      │
      ▼
FAISS Index
      │
      ▼
User Query
      │
      ▼
Query Embedding
      │
      ▼
Semantic Retrieval
      │
      ▼
Prompt Construction
      │
      ▼
Llama 3.3 (Groq)
      │
      ▼
Streaming Response
```

---

# RAG Evaluation Pipeline

The project also includes an automated evaluation pipeline to measure the quality of generated answers.

```
Evaluation Dataset
        │
        ▼
Retrieve Relevant Chunks
        │
        ▼
Generate RAG Answer
        │
        ▼
LLM-as-a-Judge Evaluation
        │
        ▼
Groundedness
Relevance
Completeness
Correctness
        │
        ▼
Average Evaluation Score
```

The evaluator compares:

* User Question
* Retrieved Context
* Generated Answer
* Ground Truth Answer

and returns structured scores along with qualitative feedback.

---

# Technologies Used

## Embedding Model

**BAAI/bge-small-en-v1.5**

* Strong semantic retrieval performance
* Lightweight and efficient
* Excellent sentence-level similarity

## Large Language Model

**llama-3.3-70b-versatile (Groq API)**

* Fast inference
* Strong reasoning capability
* High-quality instruction following
* Streaming response support

## Vector Database

**FAISS**

* Efficient similarity search
* Local vector storage
* Optimized retrieval performance

---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd AL_chatbot
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file.

```text
GROQ_API_KEY=your_api_key_here
```

---

# Running the Chatbot

```bash
streamlit run app.py
```

Steps:

1. Upload a PDF.
2. Wait for indexing.
3. Ask questions.
4. View retrieved source chunks.
5. Receive streamed responses.

---

# Running the Evaluation Pipeline

Run:

```bash
python -m src.evaluate_pipeline
```

The evaluation pipeline:

* Loads the evaluation dataset
* Retrieves relevant document chunks
* Generates answers using the RAG pipeline
* Uses Groq Llama 3.3 as an LLM Judge
* Scores each answer on:

  * Groundedness
  * Relevance
  * Completeness
  * Correctness
* Computes the average evaluation score

---

# RAG Pipeline

## Document Processing

1. PDF extraction
2. Text cleaning
3. Recursive chunking
4. Embedding generation
5. FAISS indexing

## Retrieval

1. Convert user query into an embedding.
2. Perform FAISS similarity search.
3. Retrieve the top relevant chunks.

## Generation

1. Build prompt using retrieved context.
2. Generate response with Llama 3.3.
3. Stream the response to the UI.

---

# Prompt Strategy

The chatbot is strictly grounded in retrieved context.

Rules:

* Answer only from retrieved context.
* Do not use outside knowledge.
* Do not hallucinate.
* If the answer is unavailable, respond:

```
NOT FOUND IN DOCUMENT
```

---

# Project Structure

```text
AL_chatbot/
│
├── chunks/
├── data/
├── notebooks/
├── vectordb/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py
│   ├── retriever.py
│   ├── generator.py
│   ├── evaluator.py
│   ├── evaluation_data.py
│   └── evaluate_pipeline.py
│
├── app.py
├── requirements.txt
├── README.md
└── .env
```

---

# Evaluation Metrics

The LLM evaluator scores each generated answer using:

* **Groundedness** – Is the answer supported by the retrieved context?
* **Relevance** – Does it answer the user's question?
* **Completeness** – Does it cover all important information?
* **Correctness** – Is it consistent with the expected answer?

Each metric is scored from **1–10**, along with:

* Overall score
* Feedback explaining the evaluation

---

# Limitations

* Processes one PDF at a time
* Retrieval quality depends on chunking strategy
* Large PDFs require longer indexing
* Limited to retrieved context
* LLM-based evaluation may exhibit minor subjective variation

---

# Author

**Ashutosh Narayan**
