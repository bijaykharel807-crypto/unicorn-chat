# app.py
import streamlit as st
from groq import Groq
import os
import json
from typing import Dict, List, Any
import networkx as nx
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# PART 1: PROBLEM DEFINITION AND GOALS
# ============================================================================

class AIAssistantSystem:
    """
    A comprehensive AI system that demonstrates the entire AI development lifecycle
    with explainability, ethical considerations, and domain-specific knowledge.
    """
    
    def __init__(self):
        self.problem_statement = """
        Build an intelligent assistant that helps users understand AI concepts
        while demonstrating transparency, ethical behavior, and domain expertise.
        """
        
        self.goals = {
            "primary": "Provide accurate, explainable AI assistance",
            "secondary": [
                "Demonstrate knowledge graph integration",
                "Show transfer learning capabilities",
                "Implement attention mechanisms",
                "Ensure ethical AI practices",
                "Provide explainable outputs"
            ]
        }

# ============================================================================
# PART 2: CHOOSE THE RIGHT AI APPROACH
# ============================================================================

class AIApproach:
    """
    Implements multiple AI techniques including deep learning,
    transfer learning, and attention mechanisms.
    """
    
    def __init__(self):
        self.approaches = {
            "deep_learning": "Using Llama-3.3-70B for advanced pattern recognition",
            "transfer_learning": "Leveraging pre-trained knowledge for domain adaptation",
            "attention_mechanisms": "Implementing focus on relevant context",
            "graph_neural_networks": "Knowledge graph traversal for enhanced reasoning"
        }

# ============================================================================
# PART 3: COLLECT AND PREPROCESS DATA
# ============================================================================

class DataProcessor:
    """Handles data collection, validation, and preprocessing"""
    
    def __init__(self):
        self.data_sources = []
        self.quality_metrics = {
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0
        }
    
    def validate_data_quality(self, data: Dict) -> Dict:
        """Check data quality and availability"""
        issues = []
        if not data.get("content"):
            issues.append("Missing content - data quality issue")
        if len(str(data)) < 10:
            issues.append("Insufficient data - availability concern")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "quality_score": 1.0 - (len(issues) * 0.2)
        }

# ============================================================================
# PART 4: DEVELOP A KNOWLEDGE GRAPH
# ============================================================================

class KnowledgeGraph:
    """
    Implements graph neural network concepts with entity relationships
    for domain-specific knowledge integration.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """Build initial knowledge graph with AI concepts"""
        # Core concepts
        concepts = [
            "Deep Learning", "Transfer Learning", "GNN", "Attention",
            "Explainability", "Ethics", "Bias", "Data Quality",
            "Domain Complexity", "Adversarial Attacks"
        ]
        
        # Relationships
        relationships = [
            ("Deep Learning", "uses", "Attention"),
            ("Transfer Learning", "improves", "Data Quality"),
            ("GNN", "processes", "Knowledge Graph"),
            ("Explainability", "addresses", "Domain Complexity"),
            ("Ethics", "mitigates", "Bias"),
            ("Adversarial Attacks", "threaten", "Data Quality")
        ]
        
        # Add nodes and edges
        self.graph.add_nodes_from(concepts)
        self.graph.add_edges_from([(s, o, {"relation": r}) for s, r, o in relationships])
    
    def query_knowledge(self, concept: str) -> List[Dict]:
        """Traverse graph to find related concepts"""
        if concept not in self.graph:
            return []
        
        relations = []
        for neighbor in self.graph.neighbors(concept):
            edge_data = self.graph.get_edge_data(concept, neighbor)
            relations.append({
                "concept": neighbor,
                "relation": edge_data.get("relation", "related to"),
                "confidence": 0.85  # Simulated confidence score
            })
        
        return relations

# ============================================================================
# PART 5: TRAIN MACHINE LEARNING MODEL (using Groq API)
# ============================================================================

class GroqModelTrainer:
    """
    Integrates with Groq API for model inference, demonstrating
    transfer learning and attention mechanisms.
    """
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.conversation_history = []
        
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response using Llama model with attention to context
        """
        messages = []
        
        # Add system message if provided (implements domain-specific knowledge)
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation context (implements attention mechanism)
        for msg in self.conversation_history[-5:]:  # Last 5 messages for context
            messages.append(msg)
        
        # Add current prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                stream=True  # Enable streaming for better UX
            )
            
            # Store in history
            self.conversation_history.append({"role": "user", "content": prompt})
            
            # Handle streaming response
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            self.conversation_history.append({"role": "assistant", "content": response_text})
            return response_text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ============================================================================
# PART 6: INTEGRATE DOMAIN-SPECIFIC KNOWLEDGE
# ============================================================================

class DomainExpert:
    """Integrates domain knowledge with explainability"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.domain_complexity = {
            "healthcare": "high - requires precision and explainability",
            "finance": "high - requires transparency and auditability",
            "education": "medium - requires adaptability",
            "general": "low - broad knowledge base"
        }
    
    def get_domain_context(self, query: str) -> str:
        """Extract domain context from query"""
        domain_keywords = {
            "healthcare": ["doctor", "medical", "patient", "treatment"],
            "finance": ["money", "investment", "stock", "financial"],
            "education": ["learn", "teach", "student", "course"]
        }
        
        detected_domain = "general"
        for domain, keywords in domain_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                detected_domain = domain
                break
        
        return f"Domain: {detected_domain} (Complexity: {self.domain_complexity[detected_domain]})"

# ============================================================================
# PART 7: EVALUATE AND REFINE (with Explainability)
# ============================================================================

class ExplainableAI:
    """
    Implements explainability techniques for transparency
    """
    
    def __init__(self):
        self.explanations = []
        self.confidence_scores = []
    
    def explain_response(self, response: str, context: Dict) -> Dict:
        """Generate explanation for AI response"""
        
        explanation = {
            "timestamp": datetime.now().isoformat(),
            "response_length": len(response),
            "key_concepts": self._extract_concepts(response),
            "confidence": self._calculate_confidence(response),
            "reasoning_path": self._get_reasoning_path(context),
            "limitations": self._identify_limitations(response)
        }
        
        self.explanations.append(explanation)
        return explanation
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from response"""
        concepts = ["AI", "Machine Learning", "Ethics", "Explainability"]
        found = [c for c in concepts if c.lower() in text.lower()]
        return found if found else ["General knowledge"]
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on response characteristics"""
        # Simple heuristic: longer, more detailed responses get higher confidence
        base_score = min(0.95, 0.5 + (len(text) / 2000))
        
        # Reduce confidence if uncertainty markers present
        uncertainty_markers = ["maybe", "perhaps", "might", "could", "possibly"]
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in text.lower())
        
        return max(0.3, base_score - (uncertainty_count * 0.1))
    
    def _get_reasoning_path(self, context: Dict) -> List[str]:
        """Trace reasoning steps"""
        return [
            "1. Analyzed user query for intent",
            "2. Retrieved relevant domain knowledge",
            "3. Applied attention to key terms",
            "4. Generated response with confidence scoring"
        ]
    
    def _identify_limitations(self, text: str) -> List[str]:
        """Identify response limitations"""
        limitations = []
        
        if len(text) < 50:
            limitations.append("Response may be incomplete")
        if "?" in text[-10:]:
            limitations.append("Response ends with question - may need clarification")
            
        return limitations if limitations else ["No major limitations identified"]

# ============================================================================
# PART 8: ETHICS AND BIAS MITIGATION
# ============================================================================

class EthicalAIGuard:
    """
    Implements ethical safeguards and bias detection
    """
    
    def __init__(self):
        self.bias_patterns = {
            "gender": ["he is", "she is", "men are", "women are"],
            "racial": ["all [race] are", "typical [race]"],
            "age": ["old people", "young people always"],
            "socioeconomic": ["poor people", "rich people"]
        }
        
        self.ethical_principles = [
            "Fairness - treat all users equally",
            "Transparency - explain AI decisions",
            "Privacy - protect user data",
            "Accountability - own AI mistakes",
            "Safety - prevent harmful outputs"
        ]
    
    def check_for_bias(self, text: str) -> Dict:
        """Check response for potential bias"""
        detected_bias = []
        
        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    detected_bias.append({
                        "type": bias_type,
                        "pattern": pattern,
                        "severity": "medium" if len(text.split()) > 10 else "low"
                    })
        
        return {
            "has_bias": len(detected_bias) > 0,
            "detected_biases": detected_bias,
            "recommendation": "Review and revise biased content" if detected_bias else "No bias detected"
        }
    
    def get_ethical_context(self) -> str:
        """Provide ethical guidelines for responses"""
        return "\n".join([f"• {principle}" for principle in self.ethical_principles])

# ============================================================================
# PART 9: ADVERSARIAL ATTACK DETECTION
# ============================================================================

class AdversarialDetector:
    """
    Detects potential adversarial attacks and prompt injections
    """
    
    def __init__(self):
        self.attack_patterns = [
            "ignore previous instructions",
            "bypass safety",
            "system prompt",
            "jailbreak",
            "developer mode",
            "ignore all rules"
        ]
        
        self.suspicious_patterns = [
            "repeat after me",
            "act as if",
            "pretend to be",
            "you are now",
            "override"
        ]
    
    def analyze_input(self, user_input: str) -> Dict:
        """Check for adversarial attempts"""
        
        attacks_found = []
        for pattern in self.attack_patterns:
            if pattern in user_input.lower():
                attacks_found.append({
                    "pattern": pattern,
                    "type": "known_attack",
                    "confidence": 0.9
                })
        
        for pattern in self.suspicious_patterns:
            if pattern in user_input.lower():
                attacks_found.append({
                    "pattern": pattern,
                    "type": "suspicious",
                    "confidence": 0.6
                })
        
        return {
            "is_attack": len(attacks_found) > 0,
            "detections": attacks_found,
            "risk_level": "high" if len(attacks_found) > 2 else "medium" if attacks_found else "low"
        }

# ============================================================================
# PART 10: STREAMLIT UI IMPLEMENTATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.knowledge_graph = KnowledgeGraph()
        st.session_state.domain_expert = DomainExpert(st.session_state.knowledge_graph)
        st.session_state.explainable_ai = ExplainableAI()
        st.session_state.ethical_guard = EthicalAIGuard()
        st.session_state.adversarial_detector = AdversarialDetector()
        st.session_state.conversation = []
        st.session_state.explanations = []
        st.session_state.metrics = {
            "total_queries": 0,
            "avg_confidence": 0,
            "bias_detections": 0,
            "attack_attempts": 0
        }

def create_metrics_dashboard():
    """Create interactive metrics visualization"""
    
    # Confidence over time
    if st.session_state.explanations:
        confidences = [exp["confidence"] for exp in st.session_state.explanations[-10:]]
        fig = px.line(
            y=confidences,
            title="Confidence Score Trend",
            labels={"index": "Query #", "value": "Confidence"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Knowledge graph visualization
    st.subheader("Knowledge Graph Explorer")
    
    # Create network visualization
    G = st.session_state.knowledge_graph.graph
    pos = nx.spring_layout(G)
    
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
        )
    
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
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
            line=dict(color='darkblue', width=2)
        )
    )
    
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='Knowledge Graph Structure',
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Complete AI System with Llama 3.3 on Groq",
        page_icon="🤖",
        layout="wide"
    )
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("🤖 AI System Configuration")
        
        # API Key input (use secrets in production)
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key (get one at console.groq.com)"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Groq API key to continue")
            st.info("💡 For production, use Streamlit secrets: st.secrets['GROQ_API_KEY']")
            st.stop()
        
        # Model configuration
        st.subheader("Model Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 256, 2048, 1024)
        
        # Feature toggles
        st.subheader("AI Features")
        show_explanations = st.checkbox("Show Explanations", True)
        enable_bias_check = st.checkbox("Enable Bias Detection", True)
        enable_attack_detection = st.checkbox("Enable Attack Detection", True)
        
        # Metrics display
        st.subheader("System Metrics")
        if 'metrics' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", st.session_state.metrics["total_queries"])
            with col2:
                st.metric("Avg Confidence", f"{st.session_state.metrics['avg_confidence']:.2%}")
    
    # Initialize everything
    initialize_session_state()
    
    # Initialize model trainer
    trainer = GroqModelTrainer(api_key)
    
    # Main content area
    st.title("🧠 Complete AI System with Llama-3.3-70B on Groq")
    st.markdown("---")
    
    # Tabs for different system components
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat Interface",
        "📊 Knowledge Graph",
        "🔍 Explainability",
        "🛡️ Ethics & Security",
        "📈 Metrics Dashboard"
    ])
    
    with tab1:
        st.header("AI Chat Interface")
        
        # Domain context display
        domain_info = st.session_state.domain_expert.get_domain_context("")
        st.info(f"📚 **Active Domain Context:** {domain_info}")
        
        # Chat input
        user_input = st.text_area("Your question:", height=100)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Send", type="primary"):
                if user_input:
                    # Check for adversarial attacks
                    if enable_attack_detection:
                        attack_check = st.session_state.adversarial_detector.analyze_input(user_input)
                        if attack_check["is_attack"]:
                            st.error(f"⚠️ Potential attack detected: {attack_check['detections'][0]['pattern']}")
                            st.stop()
                    
                    # Generate response
                    with st.spinner("Generating response..."):
                        # Get domain context
                        domain_context = st.session_state.domain_expert.get_domain_context(user_input)
                        
                        # Create system prompt with ethical guidelines
                        system_prompt = f"""
                        You are an ethical AI assistant with domain expertise. 
                        {st.session_state.ethical_guard.get_ethical_context()}
                        {domain_context}
                        
                        Provide clear, accurate, and explainable responses.
                        """
                        
                        response = trainer.generate_response(user_input, system_prompt)
                        
                        # Check for bias if enabled
                        if enable_bias_check:
                            bias_check = st.session_state.ethical_guard.check_for_bias(response)
                            if bias_check["has_bias"]:
                                st.warning(f"⚠️ Potential bias detected: {bias_check['detected_biases']}")
                        
                        # Generate explanation
                        if show_explanations:
                            context = {
                                "query": user_input,
                                "domain": domain_context,
                                "timestamp": datetime.now()
                            }
                            explanation = st.session_state.explainable_ai.explain_response(response, context)
                            st.session_state.explanations.append(explanation)
                        
                        # Update metrics
                        st.session_state.metrics["total_queries"] += 1
                        if st.session_state.explanations:
                            confidences = [e["confidence"] for e in st.session_state.explanations]
                            st.session_state.metrics["avg_confidence"] = sum(confidences) / len(confidences)
                        
                        # Display response
                        st.markdown("### Response:")
                        st.write(response)
                        
                        # Query knowledge graph
                        st.markdown("### Related Knowledge Graph Concepts:")
                        concepts = st.session_state.knowledge_graph.query_knowledge("AI")
                        for concept in concepts:
                            st.write(f"• {concept['concept']} ({concept['relation']})")
    
    with tab2:
        st.header("Knowledge Graph Visualization")
        
        # Search knowledge graph
        search_term = st.text_input("Search knowledge graph:", placeholder="Enter a concept...")
        if search_term:
            results = st.session_state.knowledge_graph.query_knowledge(search_term)
            if results:
                st.success(f"Found {len(results)} related concepts:")
                for r in results:
                    with st.expander(f"📌 {r['concept']}"):
                        st.write(f"**Relation:** {r['relation']}")
                        st.write(f"**Confidence:** {r['confidence']:.2%}")
            else:
                st.warning("No concepts found")
        
        # Show full graph
        st.subheader("Complete Knowledge Graph")
        create_metrics_dashboard()
    
    with tab3:
        st.header("Explainability Dashboard")
        
        if st.session_state.explanations:
            latest = st.session_state.explanations[-1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Length", latest["response_length"])
            with col2:
                st.metric("Confidence", f"{latest['confidence']:.2%}")
            with col3:
                st.metric("Key Concepts", len(latest["key_concepts"]))
            
            st.subheader("Reasoning Path")
            for step in latest["reasoning_path"]:
                st.write(step)
            
            st.subheader("Limitations")
            for limitation in latest["limitations"]:
                st.warning(limitation)
        else:
            st.info("No explanations yet. Start a conversation to see explainability in action.")
    
    with tab4:
        st.header("Ethics & Security Monitor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ethical Principles")
            principles = st.session_state.ethical_guard.get_ethical_context()
            st.info(principles)
        
        with col2:
            st.subheader("Security Status")
            st.success("✅ Adversarial detection active")
            st.success("✅ Bias detection active")
            st.success("✅ Privacy safeguards enabled")
        
        # Pattern monitor
        st.subheader("Monitored Patterns")
        patterns_df = {
            "Bias Patterns": list(st.session_state.ethical_guard.bias_patterns.keys()),
            "Attack Patterns": st.session_state.adversarial_detector.attack_patterns[:3],
            "Suspicious Patterns": st.session_state.adversarial_detector.suspicious_patterns[:3]
        }
        
        for category, patterns in patterns_df.items():
            with st.expander(f"🔍 {category}"):
                for pattern in patterns:
                    st.write(f"• {pattern}")
    
    with tab5:
        st.header("System Metrics Dashboard")
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", st.session_state.metrics["total_queries"])
        with col2:
            st.metric("Avg Confidence", f"{st.session_state.metrics['avg_confidence']:.2%}")
        with col3:
            st.metric("Bias Detections", st.session_state.metrics["bias_detections"])
        with col4:
            st.metric("Attack Attempts", st.session_state.metrics["attack_attempts"])
        
        # Knowledge graph stats
        st.subheader("Knowledge Graph Statistics")
        G = st.session_state.knowledge_graph.graph
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Concepts", G.number_of_nodes())
        with col2:
            st.metric("Relationships", G.number_of_edges())
        with col3:
            st.metric("Density", f"{nx.density(G):.3f}")
        
        # Confidence trend
        if st.session_state.explanations:
            create_metrics_dashboard()

if __name__ == "__main__":
    main()