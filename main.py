"""
main.py
--------
A practical RAG chatbot for company policies using:
- Sentence Transformers for embeddings
- FAISS for vector search
- Llama 3.3 70B (via Groq API, direct HTTP requests)
- Streamlit for the user interface
"""
import os
import pickle
import requests
import json
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Configuration ----------
POLICIES_FOLDER = "policies"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "policy_index.faiss"
CHUNKS_PATH = "chunks.pkl"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

# ---------- Cached Model Loader ----------
@st.cache_resource
def load_model():
    """Load the sentence transformer model (cached)."""
    return SentenceTransformer(EMBEDDING_MODEL)

# ---------- Document Loading & Preprocessing ----------
def load_documents(folder_path):
    """Load all PDF and TXT files from the given folder."""
    if not os.path.exists(folder_path):
        st.warning("Folder 'policies' does not exist. Please create it and add policy documents.")
        return None

    docs = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    return docs

def chunk_documents(docs):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    return [chunk.page_content for chunk in chunks]

# ---------- Build FAISS Index (if not exists) ----------
def build_index():
    """Load docs, create embeddings, and save FAISS index + chunks."""
    docs = load_documents(POLICIES_FOLDER)
    if docs is None:
        return False
    if len(docs) == 0:
        st.error("No documents found. Please add policy files to the 'policies' folder.")
        return False

    chunk_texts = chunk_documents(docs)

    # Use cached model
    embedder = load_model()

    embeddings = embedder.encode(chunk_texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunk_texts, f)

    return True

# ---------- Load Retrieval Assets (cached) ----------
@st.cache_resource
def load_retriever():
    """Load the FAISS index, chunks, and embedder (cached)."""
    # Build index if missing
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        success = build_index()
        if not success:
            st.stop()

    embedder = load_model()          # Reuse the same cached model
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return embedder, index, chunks

# ---------- Retrieval Function ----------
def retrieve(query, embedder, index, chunks, k=5):
    """Return top‑k chunk texts most similar to the query."""
    query_emb = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]

# ---------- Groq API Call (without groq client) ----------
def call_llama(messages):
    """Send a chat completion request to Groq using direct HTTP."""
    headers = {
        "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 800
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        st.error(f"Failed to parse API response: {e}")
        return None

# ---------- Streamlit User Interface ----------
def main():
    st.set_page_config(page_title="Policy Assistant", page_icon="📘")
    st.title("📘 Company Policy Assistant")
    st.markdown("Ask questions about HR, IT, and company policies.")

    # Load retriever (cached)
    embedder, index, chunks = load_retriever()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    if prompt := st.chat_input("Type your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching policies..."):
                retrieved = retrieve(prompt, embedder, index, chunks, k=5)
                context = "\n\n".join(retrieved)

                system_msg = f"""You are a helpful assistant for company policies. 
Use the following context to answer the user's question. 
If the answer is not in the context, say you don't know. 
Always cite the source document name if available.

Context:
{context}
"""
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]

            with st.spinner("Generating answer..."):
                answer = call_llama(messages)

            if answer:
                st.write(answer)
                with st.expander("📄 Sources used"):
                    for i, chunk in enumerate(retrieved):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(chunk)
                        st.divider()
                st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()