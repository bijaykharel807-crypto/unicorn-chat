#!/usr/bin/env python3
"""
app.py - Complete AI pipeline using Groq (Llama 3.3) and OpenAI (embeddings/evaluation)
with knowledge graph, optional GNN, and RAG.

Requirements:
    pip install openai networkx pandas numpy scikit-learn torch torch-geometric python-dotenv

Environment variables (set in .env file):
    GROQ_API_KEY
    OPENAI_API_KEY
"""

import os
import json
import networkx as nx
import torch
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------------------------------------------------
# 1. Load environment and initialize clients
# ----------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GROQ_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set GROQ_API_KEY and OPENAI_API_KEY in environment or .env file")

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------------------
# 2. Data loading and preprocessing
# ----------------------------------------------------------------------
def load_documents(folder_path: str = "./data") -> List[str]:
    """Load text files from folder. If folder missing or empty, return dummy documents."""
    documents = []
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    documents.append(f.read())
    if not documents:
        print("No documents found. Using dummy climate data.")
        documents = [
            "The United States emitted 5.2 billion metric tons of CO2 in 2023. The Paris Agreement aims to limit warming to 1.5°C.",
            "China is the largest emitter of greenhouse gases. It pledged carbon neutrality by 2060.",
            "Sea levels are rising due to thermal expansion and melting glaciers. Coastal erosion is accelerating.",
            "The IPCC report warns of extreme weather events. Global temperatures have risen 1.1°C above pre-industrial levels."
        ]
    return documents

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Split text into word-based chunks with overlap."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ----------------------------------------------------------------------
# 3. Knowledge graph construction via Llama 3.3 (Groq)
# ----------------------------------------------------------------------
def extract_triples(chunk: str) -> List[Tuple[str, str, str]]:
    """Use Llama 3.3 to extract (subject, relation, object) triples from text."""
    prompt = f"""Extract entities and relationships from the following text as JSON.
Entities are real-world objects like countries, organizations, events, metrics, etc.
Relationships are directed triples: (entity1, relation, entity2).
Return a JSON object with key "triples" containing a list of triples, each a list of three strings.
Only output the JSON, no other text.

Text: {chunk}"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatil",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        triples = data.get("triples", [])
        # Ensure each triple is a tuple of three strings
        valid = []
        for t in triples:
            if isinstance(t, list) and len(t) == 3:
                valid.append((str(t[0]), str(t[1]), str(t[2])))
        return valid
    except Exception as e:
        print(f"Error extracting triples: {e}")
        return []

def build_knowledge_graph(chunks: List[str]) -> nx.DiGraph:
    """Iterate over chunks and add extracted triples to a NetworkX graph."""
    G = nx.DiGraph()
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        triples = extract_triples(chunk)
        for subj, rel, obj in triples:
            G.add_node(subj)
            G.add_node(obj)
            G.add_edge(subj, obj, relation=rel)
    print(f"Knowledge graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# ----------------------------------------------------------------------
# 4. Node embeddings via OpenAI
# ----------------------------------------------------------------------
def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> torch.Tensor:
    """Get OpenAI embeddings for a list of texts, return as torch tensor."""
    if not texts:
        return torch.empty(0)
    response = openai_client.embeddings.create(
        model=model,
        input=texts
    )
    embeds = [item.embedding for item in response.data]
    return torch.tensor(embeds)

def enrich_graph_with_embeddings(G: nx.DiGraph) -> Tuple[nx.DiGraph, torch.Tensor, List[str]]:
    """Add embedding vectors as node attributes and return tensor and node list."""
    nodes = list(G.nodes)
    # Create a descriptive text for each node (can be enhanced)
    node_texts = [f"{node} is a {G.nodes[node].get('type', 'entity')}" for node in nodes]
    print("Generating node embeddings...")
    emb_tensor = get_embeddings(node_texts)
    # Store embeddings as node attributes
    for i, node in enumerate(nodes):
        G.nodes[node]["embedding"] = emb_tensor[i].tolist()
    return G, emb_tensor, nodes

# ----------------------------------------------------------------------
# 5. Optional: Simple Graph Neural Network (GNN) for link prediction
#    (This is a minimal example; training can be skipped if not needed)
# ----------------------------------------------------------------------
class SimpleGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = torch.nn.Linear(in_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def train_gnn_if_needed(G, node_embs, nodes):
    """Placeholder: train a simple node classifier or link predictor."""
    # In a real scenario, you'd use PyTorch Geometric. Here we just print.
    print("GNN training skipped (enable by implementing torch_geometric model).")
    # For demonstration, we'll just return a dummy model.
    return SimpleGNN(node_embs.shape[1], 64, 2)

# ----------------------------------------------------------------------
# 6. Retrieval-augmented generation
# ----------------------------------------------------------------------
def retrieve_subgraph(query: str, G: nx.DiGraph, node_embs: torch.Tensor, nodes: List[str], top_k: int = 5) -> nx.DiGraph:
    """Find top_k most similar nodes to query, return induced subgraph."""
    q_emb = get_embeddings([query])
    sims = torch.cosine_similarity(q_emb, node_embs)
    top_indices = sims.topk(min(top_k, len(nodes))).indices.tolist()
    relevant_nodes = [nodes[i] for i in top_indices]
    return G.subgraph(relevant_nodes).copy()

def subgraph_to_text(subgraph: nx.DiGraph) -> str:
    """Convert subgraph edges to human-readable text."""
    lines = []
    for u, v, data in subgraph.edges(data=True):
        rel = data.get("relation", "related to")
        lines.append(f"{u} -- {rel} --> {v}")
    return "\n".join(lines)

def answer_question(question: str, G: nx.DiGraph, node_embs: torch.Tensor, nodes: List[str]) -> str:
    """Retrieve relevant subgraph, feed to Llama 3.3, return answer."""
    subgraph = retrieve_subgraph(question, G, node_embs, nodes)
    context = subgraph_to_text(subgraph)

    system_msg = "You are an expert in climate science and related domains. Use the knowledge graph facts provided to answer the question. If the facts are insufficient, say you don't know."
    user_msg = f"Knowledge Graph facts:\n{context}\n\nQuestion: {question}\n\nFirst think step by step, then provide a concise answer."

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatil",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content

# ----------------------------------------------------------------------
# 7. Evaluation using OpenAI (optional)
# ----------------------------------------------------------------------
def evaluate_answer(question: str, generated: str, golden: str) -> dict:
    """Use GPT-4o-mini to score answer accuracy, completeness, conciseness."""
    prompt = f"""You are an evaluator. Rate the generated answer on three dimensions 1-5:
- Factual accuracy compared to the golden answer
- Completeness (covers all important points)
- Conciseness (no unnecessary verbosity)

Question: {question}
Golden answer: {golden}
Generated answer: {generated}

Output a JSON object with keys: accuracy, completeness, conciseness.
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {"accuracy": 0, "completeness": 0, "conciseness": 0}

# ----------------------------------------------------------------------
# 8. Safety check using Llama Guard (if available on Groq)
# ----------------------------------------------------------------------
def check_safety(text: str) -> bool:
    """Return True if text is safe."""
    try:
        response = groq_client.chat.completions.create(
            model="llama-guard-4-12b",  # Check Groq model catalog for exact name
            messages=[{"role": "user", "content": text}],
            temperature=0,
            max_tokens=10
        )
        result = response.choices[0].message.content.lower()
        return "safe" in result
    except Exception as e:
        print(f"Safety check failed: {e}")
        return True  # Assume safe if model not available

# ----------------------------------------------------------------------
# 9. Main pipeline
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("AI System: Groq (Llama 3.3) + OpenAI Embeddings + Knowledge Graph")
    print("=" * 60)

    # Step 1: Load and chunk documents
    print("\n[1] Loading documents...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    print(f"Created {len(all_chunks)} chunks.")

    # Step 2: Build knowledge graph using Llama 3.3
    print("\n[2] Building knowledge graph from chunks...")
    G = build_knowledge_graph(all_chunks[:5])  # limit for demo speed

    if G.number_of_nodes() == 0:
        print("No triples extracted. Adding dummy nodes for demonstration.")
        G.add_node("USA", type="Country")
        G.add_node("CO2", type="Metric")
        G.add_edge("USA", "CO2", relation="emits")

    # Step 3: Enrich graph with embeddings
    print("\n[3] Generating node embeddings...")
    G, node_embs, node_list = enrich_graph_with_embeddings(G)

    # Step 4: (Optional) GNN training
    print("\n[4] Initializing GNN (placeholder)...")
    gnn_model = train_gnn_if_needed(G, node_embs, node_list)

    # Step 5: Interactive Q&A
    print("\n[5] Ready for questions. Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit"):
            break

        # Safety check (optional)
        if not check_safety(question):
            print("I'm sorry, I cannot answer that question.")
            continue

        # Answer
        answer = answer_question(question, G, node_embs, node_list)
        print(f"\nAnswer: {answer}\n")

        # Optional evaluation if golden answer available (here we skip)
        # For demonstration, we could ask for golden input
        if input("Evaluate this answer? (y/n): ").lower() == "y":
            golden = input("Enter golden answer: ")
            scores = evaluate_answer(question, answer, golden)
            print("Evaluation scores:", scores)

if __name__ == "__main__":
    main()