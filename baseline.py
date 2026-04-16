"""
Baseline RAG — No graph structure, just cosine similarity retrieval.
We compare this against SubgraphRAG to show the value of graph-based retrieval.
"""
from neo4j import GraphDatabase
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import time

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"
GROQ_API_KEY   = "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"  # ← paste your gsk_ key

driver  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client  = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── STEP 1: GET ALL TRIPLES (no graph structure, flat retrieval) ──────────────
def get_all_triples(limit=500):
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE r.type IS NOT NULL
            RETURN a.name AS head, r.type AS relation, b.name AS tail
            LIMIT $limit
        """, limit=limit)
        return [{"text": f"({r['head']}, {r['relation']}, {r['tail']})"} for r in result]

# ── STEP 2: COSINE SIMILARITY RETRIEVAL (no entity anchoring) ────────────────
def baseline_retrieve(query, all_triples, top_k=50):
    texts      = [t["text"] for t in all_triples]
    query_emb  = embedder.encode(query, convert_to_numpy=True)
    triple_embs = embedder.encode(texts, convert_to_numpy=True, batch_size=64)

    query_norm   = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    triple_norms = triple_embs / (np.linalg.norm(triple_embs, axis=1, keepdims=True) + 1e-10)
    scores       = triple_norms @ query_norm

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [all_triples[i] for i in top_idx]

# ── STEP 3: LLM REASONING ────────────────────────────────────────────────────
def ask_llm(query, triples):
    triple_text = "\n".join([t["text"] for t in triples])
    prompt = f"""You are a medical assistant. Use ONLY the triples below to answer.
Reason step by step. End with answers prefixed with "ans:".
If answer not found, say "ans: not available".

Triples:
{triple_text}

Question: {query}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ── BASELINE PIPELINE ─────────────────────────────────────────────────────────
def run_baseline(query, all_triples):
    print(f"\n{'='*60}")
    print(f"❓ Question: {query}")
    print('='*60)
    triples = baseline_retrieve(query, all_triples)
    print(f"📦 Retrieved {len(triples)} triples (cosine similarity)")
    answer = ask_llm(query, triples)
    print(f"\n🤖 Baseline Answer:\n{answer}")
    return answer

# ── RUN BASELINE ON SAME 10 QUESTIONS ────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What drugs treat Alzheimer disease?",
        "What are the contraindications of Codeine?",
        "What drugs treat Parkinson disease?",
        "What are the side effects of Galantamine?",
        "What drugs treat hypertension?",
        "What are the contraindications of Metformin?",
        "What drugs treat epilepsy?",
        "What are the side effects of Donepezil?",
        "What drugs treat asthma?",
        "What are the contraindications of Aspirin?"
    ]

    print("Loading all triples for baseline...")
    all_triples = get_all_triples(limit=500)
    print(f"Loaded {len(all_triples)} triples")

    import json
    results = []
    for i, q in enumerate(questions):
        print(f"\n🧪 Baseline Test {i+1}/10")
        answer = run_baseline(q, all_triples)
        results.append({"question": q, "answer": answer})
        time.sleep(10)

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Baseline evaluation done! Saved to baseline_results.json")