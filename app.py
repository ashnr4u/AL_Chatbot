# app.py
import os
import warnings
import streamlit as st
from dotenv import load_dotenv

# Suppress warnings bcz of hugging face
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Import from src modules
from src.config import LLM_MODEL, EMBEDDING_MODEL, DATA_PATH
from src.retriever import load_embedding_model, retrieve
from src.generator import stream_response
from src.pipeline import process_pdf

# Load environment variables
load_dotenv()


#------------setting streamlit------------------------

#for ui/ux
st.set_page_config(
    page_title="A.Lab's Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("A.Lab's Chatbot")

# chat history persists
# UI can re-render old messages
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False


#---------SIDEBAR---------------

with st.sidebar:
    st.header("System Information")
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    # Check for existing PDF in /data folder
    if not uploaded_file and os.path.exists(DATA_PATH):
        data_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        if data_files and not st.session_state.vector_ready:
            st.info(f"Found document in /data: {data_files[0]}")
    
    st.divider()
    
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.write(f"**LLM:** {LLM_MODEL}")
    st.write(f"**Embedding Model:** {EMBEDDING_MODEL}")
    
    if "chunks" in st.session_state:
        st.write(f"**Chunks Indexed:** {len(st.session_state.chunks)}")

# ==========================================================
# PDF INDEXING
# ==========================================================

if not st.session_state.vector_ready:
    pdf_to_process = None
    
    if uploaded_file is not None:
        pdf_to_process = uploaded_file
    elif os.path.exists(DATA_PATH):
        data_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        if data_files:
            pdf_to_process = data_files[0]  # Pass filename, will be handled in process_pdf
    
    if pdf_to_process:
        with st.spinner("Processing PDF and creating embeddings..."):
            
            try:
                if uploaded_file:
                    embedding_model, chunks, index = process_pdf( uploaded_pdf=uploaded_file)
                else:
                    embedding_model, chunks, index = process_pdf(pdf_path=os.path.join(DATA_PATH, pdf_to_process))
            except ValueError as e:
                st.error(f"PDF Processing Failed: {e}")
                st.stop()
            st.session_state.embedding_model = embedding_model
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.vector_ready = True
        
        st.success(f"PDF indexed successfully! {len(chunks)} chunks created.")
        st.rerun()

# ==========================================================
# DISPLAY CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================================
# CHAT INPUT WITH TRUE TOKEN-BY-TOKEN STREAMING
# ==========================================================

if st.session_state.vector_ready:
    query = st.chat_input("Ask a question about the document...")
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            # Retrieve chunks
            retrieved_chunks = retrieve(
                query=query,
                embedding_model=st.session_state.embedding_model,
                index=st.session_state.index,
                chunks=st.session_state.chunks
            )
            
            context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
            
            #streaming token by token
            placeholder = st.empty()
            answer = ""
            
            for token in stream_response(query, context):
                answer += token
                placeholder.markdown(answer + "▌")
            
            # Remove cursor at the end
            placeholder.markdown(answer)
            
            # Display source chunks
            if retrieved_chunks:
                st.markdown("---")
                st.markdown("### Source Chunks")
                
                for i, source in enumerate(retrieved_chunks, start=1):
                    with st.expander(f"Chunk {i} | Page {source['page']} | Score {source['score']:.3f}"):
                        st.write(source["text"])
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload a PDF or place a PDF in the /data folder to start chatting.")