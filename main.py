"""
General-Purpose AI Assistant using Llama 3.3-70B (Groq) + Streamlit

Features:
- Upload documents (PDF) for context-aware Q&A (RAG with FAISS)
- Automatic knowledge graph construction (entity-relation triples)
- Interactive graph visualization (Plotly)
- Graph reasoning simulation (entity extraction + LLM reasoning)
- Explainability (step-by-step explanations)
- Safety & bias checks via prompt-based moderation
- Fully customizable for any domain

Install required packages:
    pip install streamlit groq langchain langchain-community faiss-cpu sentence-transformers pypdf networkx pandas plotly

Create .streamlit/secrets.toml with:
    GROQ_API_KEY = "your-api-key"
"""

import streamlit as st
from groq import Groq
import networkx as nx
import plotly.graph_objects as go
from pypdf import PdfReader
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ------------------------------
# Page configuration
st.set_page_config(page_title="General AI Assistant", layout="wide")
st.title("🧠 General AI Assistant (Llama 3.3-70B + Groq)")

# ------------------------------
# Initialize Groq client
if "groq_client" not in st.session_state:
    st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

client = st.session_state.groq_client

# ------------------------------
# Session state for data persistence
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "graph" not in st.session_state:
    st.session_state.graph = nx.MultiDiGraph()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Default system prompt (can be customized in sidebar)
DEFAULT_SYSTEM_PROMPT = """You are a helpful, knowledgeable AI assistant. Answer questions based on the provided context when available. If the context doesn't contain the answer, say you don't know. Be concise and accurate."""

# ------------------------------
# Helper functions

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

@st.cache_resource
def get_embeddings_model():
    """Load sentence transformer embeddings model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vectorstore(chunks):
    """Create FAISS vector store from text chunks."""
    embeddings = get_embeddings_model()
    return FAISS.from_texts(chunks, embeddings)

def extract_triples(text_chunk):
    """Use Llama to extract (entity, relation, entity) triples from text (domain-agnostic)."""
    prompt = f"""Extract all knowledge triples (entity1, relation, entity2) from the following text. 
Return each triple on a new line, formatted as: entity1 | relation | entity2

Text: {text_chunk}

Triples:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        content = response.choices[0].message.content.strip()
        triples = []
        for line in content.split("\n"):
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    triples.append(tuple(parts))
        return triples
    except Exception as e:
        st.error(f"Triple extraction failed: {e}")
        return []

def build_graph_from_chunks(chunks):
    """Extract triples from each chunk and add to NetworkX graph."""
    graph = nx.MultiDiGraph()
    progress_bar = st.progress(0, text="Extracting knowledge triples...")
    for i, chunk in enumerate(chunks):
        triples = extract_triples(chunk)
        for subj, rel, obj in triples:
            graph.add_node(subj, type="entity")
            graph.add_node(obj, type="entity")
            graph.add_edge(subj, obj, relation=rel)
        progress_bar.progress((i + 1) / len(chunks))
    progress_bar.empty()
    return graph

def plot_graph(graph):
    """Create an interactive Plotly graph visualization."""
    if graph.number_of_nodes() == 0:
        return None
    pos = nx.spring_layout(graph, seed=42)
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(color='midnightblue', width=1)
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    return fig

def retrieve_context(question, k=3):
    """Retrieve relevant chunks from vector store."""
    if st.session_state.vectorstore is None:
        return ""
    docs = st.session_state.vectorstore.similarity_search(question, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def answer_question(question, context, system_prompt):
    """Answer question using retrieved context and custom system prompt."""
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

def explain_answer(question, answer, context, system_prompt):
    """Generate step-by-step explanation of how the answer was derived."""
    prompt = f"""Explain step-by-step how the answer was derived, referencing the context.

Question: {question}
Answer: {answer}
Context: {context}

Explanation:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Explanation unavailable: {e}"

def is_safe_input(text):
    """Check if user input is safe (no harmful/adversarial content)."""
    prompt = f"""Classify the following user input as SAFE or UNSAFE. 
UNSAFE if it contains harmful, unethical, adversarial, or offensive content.

Input: {text}

Classification (SAFE or UNSAFE):"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip().upper()
        return "SAFE" in result
    except:
        return True  # Default to safe if API fails

def check_bias(text):
    """Check if a response contains bias (gender, racial, cultural, etc.)."""
    prompt = f"""Does the following response contain any gender, racial, cultural, or other form of bias? 
Answer YES or NO and provide a brief explanation.

Response: {text}

Bias analysis:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Bias check failed: {e}"

def decompose_query(query):
    """Break a complex query into simpler sub-questions."""
    prompt = f"Break down this complex question into simpler sub-questions (one per line):\n\n{query}"
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return [q.strip() for q in response.choices[0].message.content.split("\n") if q.strip()]
    except:
        return [query]

def graph_reasoning(question, system_prompt):
    """Use the knowledge graph to answer a question (simulate GNN)."""
    # Extract entities from question
    entity_prompt = f"Extract the main entities from this question. Return them as a comma-separated list: {question}"
    try:
        entities_str = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": entity_prompt}],
            temperature=0.1,
            max_tokens=50
        ).choices[0].message.content
        entities = [e.strip() for e in entities_str.split(",") if e.strip()]
    except:
        entities = []

    # Find entities in graph
    relevant_nodes = [e for e in entities if e in st.session_state.graph]
    if not relevant_nodes:
        return "No relevant entities found in knowledge graph."

    # Build subgraph description
    subgraph_info = ""
    for node in relevant_nodes:
        neighbors = list(st.session_state.graph.neighbors(node))
        if neighbors:
            subgraph_info += f"{node} is connected to: {', '.join(neighbors)}\n"
        else:
            subgraph_info += f"{node} has no outgoing connections.\n"

    # Reason over subgraph
    reasoning_prompt = f"""Based on the following knowledge graph connections, answer the question.

Graph connections:
{subgraph_info}

Question: {question}

Answer:"""
    try:
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": reasoning_prompt}],
            temperature=0.2,
            max_tokens=500
        ).choices[0].message.content
    except Exception as e:
        return f"Graph reasoning failed: {e}"

# ------------------------------
# Sidebar: Configuration and document upload
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Custom system prompt
    system_prompt = st.text_area(
        "System Prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=150,
        help="Customize the AI's behavior and expertise domain."
    )
    
    st.divider()
    
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)
            if text:
                st.session_state.chunks = chunk_text(text)
                st.success(f"Extracted {len(st.session_state.chunks)} chunks.")
                
                # Build vector store
                with st.spinner("Creating vector store..."):
                    st.session_state.vectorstore = create_vectorstore(st.session_state.chunks)
                    st.success("Vector store ready.")
                
                # Build knowledge graph
                with st.spinner("Building knowledge graph (this may take a while)..."):
                    st.session_state.graph = build_graph_from_chunks(st.session_state.chunks)
                    st.success(f"Graph built with {st.session_state.graph.number_of_nodes()} nodes and {st.session_state.graph.number_of_edges()} edges.")
            else:
                st.error("No text could be extracted from the PDF.")
    
    st.divider()
    st.header("🧪 Advanced Options")
    st.session_state.use_graph_reasoning = st.checkbox(
        "Use knowledge graph reasoning (if available)",
        value=False,
        help="When enabled, the assistant will try to answer using the extracted graph before falling back to RAG."
    )

# ------------------------------
# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🔗 Knowledge Graph", "🔍 Explainability", "🛡️ Admin"])

# ----- Tab 1: Chat -----
with tab1:
    st.subheader("Ask a question")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    query = st.chat_input("Type your question here...")
    
    if query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Safety check
        if not is_safe_input(query):
            response = "I'm sorry, but your input appears unsafe. Please rephrase."
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.error(response)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context = retrieve_context(query) if st.session_state.vectorstore else ""
                    
                    # Decide between graph reasoning and RAG
                    if st.session_state.use_graph_reasoning and st.session_state.graph.number_of_nodes() > 0:
                        answer = graph_reasoning(query, system_prompt)
                    else:
                        if not context and not st.session_state.vectorstore:
                            answer = answer_question(query, "No context provided.", system_prompt)
                        else:
                            answer = answer_question(query, context, system_prompt)
                    
                    # Generate explanation (stored but not shown by default)
                    explanation = explain_answer(query, answer, context, system_prompt) if context else "No context available."
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Optional explanation expander
                    with st.expander("Show explanation"):
                        st.write(explanation)
                    
                    # Bias check
                    if st.button("Check for bias", key=f"bias_{len(st.session_state.chat_history)}"):
                        bias_report = check_bias(answer)
                        st.info(bias_report)
                    
                    # Feedback
                    feedback = st.feedback("thumbs", key=f"fb_{len(st.session_state.chat_history)}")
                    if feedback is not None:
                        st.caption("Thank you for your feedback!")
            
            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ----- Tab 2: Knowledge Graph -----
with tab2:
    st.subheader("Knowledge Graph Visualization")
    if st.session_state.graph.number_of_nodes() > 0:
        fig = plot_graph(st.session_state.graph)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Graph is empty.")
    else:
        st.info("No knowledge graph available. Upload a document first.")
    
    # Show graph statistics
    if st.session_state.graph.number_of_nodes() > 0:
        st.write(f"**Nodes:** {st.session_state.graph.number_of_nodes()}")
        st.write(f"**Edges:** {st.session_state.graph.number_of_edges()}")
        
        # List nodes and edges
        with st.expander("Show nodes"):
            st.write(list(st.session_state.graph.nodes()))
        with st.expander("Show edges (first 20)"):
            edges = list(st.session_state.graph.edges(data=True))[:20]
            for u, v, d in edges:
                st.write(f"{u} --({d.get('relation','connected')})--> {v}")

# ----- Tab 3: Explainability -----
with tab3:
    st.subheader("Explainability Sandbox")
    st.write("Test how the model explains its answers with custom inputs.")
    
    col1, col2 = st.columns(2)
    with col1:
        test_question = st.text_area("Enter a test question", height=100)
    with col2:
        test_context = st.text_area("Enter context (optional)", height=100, 
                                     placeholder="If left empty, a default context will be used.")
    
    if st.button("Generate Explanation"):
        if not test_question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating..."):
                # Use provided context or retrieve from vectorstore
                if not test_context and st.session_state.vectorstore:
                    test_context = retrieve_context(test_question)
                elif not test_context:
                    test_context = "No context provided."
                
                # Get answer
                answer = answer_question(test_question, test_context, system_prompt)
                explanation = explain_answer(test_question, answer, test_context, system_prompt)
                
                st.markdown("**Answer:**")
                st.info(answer)
                st.markdown("**Explanation:**")
                st.success(explanation)
                
                # Bias check
                if st.button("Check bias on this answer"):
                    bias = check_bias(answer)
                    st.write(bias)

# ----- Tab 4: Admin -----
with tab4:
    st.subheader("Admin: Safety & Bias Checks")
    
    st.markdown("### Input Safety Test")
    input_text = st.text_input("Enter text to test for safety:")
    if st.button("Check Safety"):
        if input_text:
            safe = is_safe_input(input_text)
            if safe:
                st.success("SAFE")
            else:
                st.error("UNSAFE")
    
    st.markdown("### Bias Detection Test")
    bias_text = st.text_area("Enter text to check for bias:")
    if st.button("Check Bias"):
        if bias_text:
            result = check_bias(bias_text)
            st.write(result)
    
    st.markdown("### Reset Application")
    if st.button("Clear All Data (Session Reset)"):
        for key in ["chunks", "graph", "vectorstore", "chat_history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Session cleared. Refresh the page if needed.")
        st.rerun()

# ------------------------------
# Footer
st.divider()
st.caption("Powered by Llama 3.3-70B via Groq and Streamlit. Customize the system prompt to adapt to any domain.")