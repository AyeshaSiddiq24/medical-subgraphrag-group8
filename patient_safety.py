"""
Patient Safety Checker
Given a patient's conditions, checks which drugs are SAFE vs UNSAFE for them
using the medical knowledge graph in Neo4j.
"""

from neo4j import GraphDatabase
from groq import Groq

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"
GROQ_API_KEY   = "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"  # ← paste your gsk_ key

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ── STEP 1: GET DRUGS CONTRAINDICATED FOR A CONDITION ────────────────────────
def get_unsafe_drugs(condition: str) -> list:
    """Get all drugs contraindicated for a given condition."""
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(b.name) CONTAINS toLower($condition)
            AND r.type = 'contraindication'
            RETURN a.name AS drug
            LIMIT 50
        """, condition=condition)
        return [r["drug"] for r in result if r["drug"]]

# ── STEP 2: GET SAFE DRUGS (indicated for condition) ─────────────────────────
def get_safe_drugs(condition: str) -> list:
    """Get all drugs indicated for a given condition."""
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(b.name) CONTAINS toLower($condition)
            AND r.type = 'indication'
            RETURN a.name AS drug
            LIMIT 50
        """, condition=condition)
        return [r["drug"] for r in result if r["drug"]]

# ── STEP 3: CHECK IF SPECIFIC DRUG IS SAFE FOR PATIENT ───────────────────────
def check_drug_for_patient(drug: str, conditions: list) -> dict:
    """Check if a specific drug is safe given a list of patient conditions."""
    unsafe_for = []
    for condition in conditions:
        unsafe = get_unsafe_drugs(condition)
        if any(drug.lower() in u.lower() for u in unsafe):
            unsafe_for.append(condition)
    return {
        "drug": drug,
        "safe": len(unsafe_for) == 0,
        "unsafe_for": unsafe_for
    }

# ── STEP 4: FULL PATIENT PROFILE ANALYSIS ────────────────────────────────────
def analyze_patient(conditions: list, drugs_to_check: list, client=None) -> dict:
    """
    Given patient conditions and drugs to check,
    return safety analysis for each drug.
    """
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)

    results = []
    for drug in drugs_to_check:
        result = check_drug_for_patient(drug, conditions)
        results.append(result)

    # Get drugs contraindicated for ALL patient conditions
    all_unsafe = set()
    all_safe   = set()
    for condition in conditions:
        unsafe = get_unsafe_drugs(condition)
        safe   = get_safe_drugs(condition)
        all_unsafe.update(unsafe)
        all_safe.update(safe)

    # LLM summary
    safe_results   = [r for r in results if r["safe"]]
    unsafe_results = [r for r in results if not r["safe"]]

    prompt = f"""You are a clinical pharmacology assistant.
A patient has the following conditions: {', '.join(conditions)}.

Drug safety analysis from knowledge graph:
UNSAFE drugs (contraindicated): {', '.join([r['drug'] + ' (unsafe for: ' + ', '.join(r['unsafe_for']) + ')' for r in unsafe_results]) or 'none'}
SAFE drugs (no contraindications found): {', '.join([r['drug'] for r in safe_results]) or 'none'}

Provide a brief clinical summary of:
1. Which drugs to AVOID and why
2. Which drugs appear safer based on KG data
3. Key recommendation

Keep it concise and clinical. Use only the KG data provided."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    summary = response.choices[0].message.content.strip()

    return {
        "conditions": conditions,
        "drug_results": results,
        "all_unsafe_count": len(all_unsafe),
        "all_safe_count": len(all_safe),
        "summary": summary
    }

# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    conditions    = ["diabetes", "kidney disease"]
    drugs_to_check = ["Metformin", "Codeine", "Lisinopril", "Aspirin"]

    print(f"Patient conditions: {conditions}")
    print(f"Checking drugs: {drugs_to_check}\n")

    result = analyze_patient(conditions, drugs_to_check)

    print("Drug Safety Results:")
    for r in result["drug_results"]:
        status = "✅ SAFE" if r["safe"] else f"❌ UNSAFE (for: {', '.join(r['unsafe_for'])})"
        print(f"  {r['drug']}: {status}")

    print(f"\n🤖 Clinical Summary:\n{result['summary']}")