"""
3-Hop Reasoning
Answers questions requiring 3 steps of reasoning through the KG.
Example: Disease → Drug → Side Effect → Related Disease
"""

from neo4j import GraphDatabase
from groq import Groq
import time

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"
GROQ_API_KEY   = "YOUR_GROQ_API_KEY_HERE"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = Groq(api_key=GROQ_API_KEY)

# ── 3-HOP RETRIEVAL ───────────────────────────────────────────────────────────
def get_3hop_triples(entity: str) -> list:
    """Get triples up to 3 hops away from the entity."""
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (a)-[r1]->(b)-[r2]->(c)-[r3]->(d)
            WHERE toLower(a.name) CONTAINS toLower($entity)
            AND r1.type IS NOT NULL
            AND r2.type IS NOT NULL
            AND r3.type IS NOT NULL
            RETURN
                a.name AS n1, r1.type AS r1,
                b.name AS n2, r2.type AS r2,
                c.name AS n3, r3.type AS r3,
                d.name AS n4
            LIMIT 80
        """, entity=entity)

        triples = []
        seen = set()
        for row in result:
            if None not in [row["n1"], row["r1"], row["n2"],
                            row["r2"], row["n3"], row["r3"], row["n4"]]:
                # Break into individual triples
                t1 = f"({row['n1']}, {row['r1']}, {row['n2']})"
                t2 = f"({row['n2']}, {row['r2']}, {row['n3']})"
                t3 = f"({row['n3']}, {row['r3']}, {row['n4']})"
                for t in [t1, t2, t3]:
                    if t not in seen:
                        seen.add(t)
                        triples.append({"text": t})
        return triples

# ── LLM REASONING ────────────────────────────────────────────────────────────
def ask_llm_3hop(query: str, triples: list) -> str:
    triple_text = "\n".join([t["text"] for t in triples])
    prompt = f"""You are a medical assistant reasoning over a knowledge graph.
Use ONLY the triples below to answer the question.
This question requires multiple reasoning steps (up to 3 hops).
Show your reasoning chain step by step.
End each final answer with "ans:".
If not found say "ans: not available".

Triples:
{triple_text}

Question: {query}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ── FULL 3-HOP PIPELINE ───────────────────────────────────────────────────────
def run_3hop(query: str, entity: str) -> dict:
    print(f"\n{'='*65}")
    print(f"🔗 3-Hop Query: {query}")
    print(f"📍 Anchor Entity: {entity}")
    print('='*65)

    triples = get_3hop_triples(entity)
    print(f"📦 Retrieved {len(triples)} 3-hop triples")

    answer = ask_llm_3hop(query, triples)
    print(f"\n🤖 Answer:\n{answer}")

    return {"query": query, "entity": entity, "triples": triples, "answer": answer}

# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        {
            "query": "What diseases are related to side effects of drugs that treat Alzheimer disease?",
            "entity": "Alzheimer disease"
        },
        {
            "query": "What diseases share contraindications with drugs that treat Parkinson disease?",
            "entity": "Parkinson disease"
        },
        {
            "query": "What are the side effects of drugs contraindicated for patients with epilepsy?",
            "entity": "epilepsy"
        },
    ]

    for i, q in enumerate(questions):
        run_3hop(q["query"], q["entity"])
        if i < len(questions) - 1:
            time.sleep(10)