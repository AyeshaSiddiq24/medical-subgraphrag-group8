from neo4j import GraphDatabase
from groq import Groq
import json

# ── CONFIG ────────────────────────────────────────────────────────────────────
NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"
GROQ_API_KEY   = "YOUR_GROQ_API_KEY_HERE"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = Groq(api_key=GROQ_API_KEY)

# ── STEP 1: ENTITY EXTRACTION ─────────────────────────────────────────────────
def extract_entities(query, client=None):
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""Extract all medical entities from this question.
Return ONLY a JSON list. Example: ["Hypertension", "Lisinopril"]
Question: {query}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── STEP 2: SUBGRAPH RETRIEVAL ────────────────────────────────────────────────
def get_triples(entity):
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(a.name) CONTAINS toLower($entity)
               OR toLower(b.name) CONTAINS toLower($entity)
            RETURN a.name AS head, r.type AS relation, b.name AS tail
            ORDER BY CASE r.type
                WHEN 'indication' THEN 1
                WHEN 'side_effect' THEN 2
                WHEN 'contraindication' THEN 3
                ELSE 4 END
            LIMIT 100
        """, entity=entity)
        return [{"text": f"({r['head']}, {r['relation']}, {r['tail']})"} for r in result]

# ── STEP 3: LLM REASONING ─────────────────────────────────────────────────────
def ask_llm(query, triples, client=None):
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)
    triple_text = "\n".join([t["text"] for t in triples])
    prompt = f"""You are a medical assistant. Use ONLY the triples below to answer the question.
Reason step by step. End with answers prefixed with "ans:".
If the answer is not in the triples, say "ans: not available".

Triples:
{triple_text}

Question: {query}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ── HELPER: RETRIEVE + ANSWER ─────────────────────────────────────────────────
def retrieve_subgraph_and_answer(query, client=None):
    """Returns (triples, answer) for a given query."""
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)
    entities = extract_entities(query, client)
    triples = []
    for e in entities:
        triples.extend(get_triples(e))
    answer = ask_llm(query, triples, client)
    return triples, answer

# ── FULL PIPELINE ─────────────────────────────────────────────────────────────
def run(query):
    print(f"\n{'='*60}")
    print(f"❓ Question: {query}")
    print('='*60)

    entities = extract_entities(query)
    print(f"🔍 Entities: {entities}")

    all_triples = []
    for entity in entities:
        triples = get_triples(entity)
        all_triples.extend(triples)
    print(f"📦 Retrieved {len(all_triples)} triples")

    answer = ask_llm(query, all_triples)
    print(f"\n🤖 Answer:\n{answer}")
    return answer

# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run("What drugs treat Alzheimer disease?")
    run("What are the contraindications of Codeine?")