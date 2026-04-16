from neo4j import GraphDatabase
from groq import Groq
import json
import time

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"
GROQ_API_KEY   = "YOUR_GROQ_API_KEY_HERE"  # ← paste your YOUR_GROQ_API_KEY_HERE key

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = Groq(api_key=GROQ_API_KEY)

# ── 2-HOP RETRIEVAL ───────────────────────────────────────────────────────────
def get_2hop_triples(entity):
    """Get triples 2 hops away from the entity."""
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (a)-[r1]->(b)-[r2]->(c)
            WHERE toLower(a.name) CONTAINS toLower($entity)
            RETURN a.name AS n1, r1.type AS r1, 
                   b.name AS n2, r2.type AS r2, 
                   c.name AS n3
            LIMIT 100
        """, entity=entity)
        triples = []
        for r in result:
            if None not in [r["n1"], r["r1"], r["n2"], r["r2"], r["n3"]]:
                triples.append({
                    "text": f"({r['n1']}, {r['r1']}, {r['n2']}) AND ({r['n2']}, {r['r2']}, {r['n3']})"
                })
        return triples

# ── LLM REASONING ─────────────────────────────────────────────────────────────
def ask_llm(query, triples):
    triple_text = "\n".join([t["text"] for t in triples])
    prompt = f"""You are a medical assistant. Use ONLY these triples to answer.
Reason step by step across multiple hops.
End with answers prefixed with "ans:".
If not found, say "ans: not available".

Triples:
{triple_text}

Question: {query}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ── MULTI-HOP QUESTIONS ───────────────────────────────────────────────────────
multihop_questions = [
    {
        "question": "What are the side effects of drugs that treat Alzheimer disease?",
        "entity": "Alzheimer disease",
        "hops": 2
    },
    {
        "question": "What are the side effects of drugs that treat Parkinson disease?",
        "entity": "Parkinson disease",
        "hops": 2
    },
    {
        "question": "What are the contraindications of drugs that treat epilepsy?",
        "entity": "Epilepsy",
        "hops": 2
    },
    {
        "question": "What are the side effects of drugs that treat hypertension?",
        "entity": "Hypertension",
        "hops": 2
    },
    {
        "question": "What are the side effects of drugs that treat asthma?",
        "entity": "Asthma",
        "hops": 2
    }
]

# ── RUN ───────────────────────────────────────────────────────────────────────
results = []
for i, q in enumerate(multihop_questions):
    print(f"\n{'='*60}")
    print(f"🔗 Multi-hop Test {i+1}/5 ({q['hops']} hops)")
    print(f"❓ {q['question']}")
    print('='*60)

    triples = get_2hop_triples(q["entity"])
    print(f"📦 Retrieved {len(triples)} 2-hop triples")

    answer = ask_llm(q["question"], triples)
    print(f"\n🤖 Answer:\n{answer}")

    results.append({"question": q["question"], "answer": answer})
    time.sleep(10)

with open("multihop_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Multi-hop evaluation done! Saved to multihop_results.json")
driver.close()