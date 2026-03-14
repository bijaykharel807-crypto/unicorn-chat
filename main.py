"""
Simplified AI Assistant with RAG (Llama 3.3-70B via Groq) + Streamlit
- Upload PDFs
- Ask questions based on document content
- No knowledge graph, no bias checks, no admin tab
"""

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Page config
st.set_page_config(page_title="Simple RAG Assistant", layout="wide")
st.title("📄 Simple RAG Assistant (Llama 3.3-70B + Groq)")

# Initialize Groq client
if "groq_client" not in st.session_state:
    st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
client = st.session_state.groq_client

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Helper functions ----------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page_text := page.extract_text():
            text += page_text
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vectorstore(chunks):
    embeddings = get_embeddings()
    return FAISS.from_texts(chunks, embeddings)

def retrieve_context(question, k=3):
    if st.session_state.vectorstore is None:
        return ""
    docs = st.session_state.vectorstore.similarity_search(question, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def answer_question(question, context):
    system_prompt = "You are a helpful assistant. Answer based on the provided context. If the context doesn't contain the answer, say you don't know."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
    ]
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing..."):
            text = extract_text_from_pdf(uploaded_file)
            if text:
                chunks = chunk_text(text)
                st.session_state.vectorstore = create_vectorstore(chunks)
                st.success(f"Document processed! Ready to answer questions.")
            else:
                st.error("Could not extract text from PDF.")

# ---------- Main chat ----------
st.subheader("Ask a question about your document")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Type your question...")
if query:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = retrieve_context(query) if st.session_state.vectorstore else ""
            if not context:
                answer = "Please upload a document first."
            else:
                answer = answer_question(query, context)
            st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})