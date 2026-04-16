# Medical SubgraphRAG — Group 8
### DAMG 7374 · Spring 2026 · Northeastern University

## Team Members
- Ayesha Siddiq [002319519]
- Shamamah Firdous [002058858]
- Smit Vaddoriya [002065280]

## Project Description
We adapted SubgraphRAG (ICLR 2025, Georgia Tech) to Precision Medicine using PrimeKG — a biomedical knowledge graph with 31,388 nodes and 284,089 relationships loaded into Neo4j. Our system uses Directional Distance Encoding (DDE) and a lightweight MLP with weak supervision to retrieve clinically relevant triples from the graph, achieving 60% Hit@1 on 30 medical questions compared to 20% for the cosine baseline — a 3x improvement. We extended the framework with 4 clinical tools: Drug-Drug Interaction Checker, Patient Safety Checker, Disease Explorer, and a KG Statistics Dashboard, all accessible through a live Streamlit application.

## Key Results
- SubgraphRAG Hit@1: 18/30 (60%) vs Baseline: 6/30 (20%)
- DDE+MLP indication triples in top-10: 10/10 vs Cosine: 0/10
- Overall improvement: 3x better than baseline
- Hallucination Score: 100/100 (fully grounded answers)

## How to Run
1. Install Neo4j Desktop and start the database on neo4j://127.0.0.1:7687
2. Install dependencies:
   pip3 install streamlit groq neo4j sentence-transformers torch networkx pyvis pandas
3. Add your Groq API key in app.py and pipeline.py
4. Run: python3 -m streamlit run app.py

## Key Files
- app.py — Main Streamlit application (6 tabs)
- pipeline.py — Core SubgraphRAG pipeline
- mlp_retriever.py — DDE + MLP retrieval with query-aware weak supervision
- dde.py — Directional Distance Encoding implementation
- drug_interaction.py — Drug-Drug Interaction Checker
- patient_safety.py — Patient Safety Checker
- evaluation.py — 30-question evaluation script
- baseline.py — Baseline cosine similarity evaluation
- compare_results.py — Comparison with baseline

## Architecture
User Query → Entity Extraction (Llama 3.3 70B) → Neo4j Subgraph Retrieval → DDE + MLP Scoring → LLM Reasoning → Grounded Answer

## Reference
Li, Miao & Li. Simple is Effective: The Roles of Graphs and LLMs in Knowledge Graph-based Retrieval Augmented Generation. ICLR 2025.
