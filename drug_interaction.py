"""
Drug-Drug Interaction Checker
Given two drugs, checks if they share contraindicated diseases
or have conflicting mechanisms using the medical KG in Neo4j.
"""

from neo4j import GraphDatabase
from groq import Groq
import json

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"
GROQ_API_KEY   = "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"  # ← paste your gsk_ key

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = Groq(api_key=GROQ_API_KEY)

# ── STEP 1: GET DRUG PROFILE FROM NEO4J ──────────────────────────────────────
def get_drug_profile(drug_name: str) -> dict:
    """Get all contraindications, side effects, and indications for a drug."""
    with driver.session(database="neo4j") as session:

        # Contraindications
        contra = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(a.name) CONTAINS toLower($drug)
            AND r.type = 'contraindication'
            RETURN b.name AS entity
            LIMIT 50
        """, drug=drug_name)
        contraindications = [r["entity"] for r in contra if r["entity"]]

        # Side effects
        effects = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(a.name) CONTAINS toLower($drug)
            AND r.type IN ['side_effect', 'drug_effect']
            RETURN b.name AS entity
            LIMIT 50
        """, drug=drug_name)
        side_effects = [r["entity"] for r in effects if r["entity"]]

        # Indications (what it treats)
        indications = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(a.name) CONTAINS toLower($drug)
            AND r.type = 'indication'
            RETURN b.name AS entity
            LIMIT 20
        """, drug=drug_name)
        treats = [r["entity"] for r in indications if r["entity"]]

    return {
        "drug": drug_name,
        "contraindications": contraindications,
        "side_effects": side_effects,
        "treats": treats
    }

# ── STEP 2: FIND SHARED RISKS ─────────────────────────────────────────────────
def find_shared_risks(profile1: dict, profile2: dict) -> dict:
    """Find overlapping contraindications and side effects between two drugs."""
    shared_contra = list(set(profile1["contraindications"]) &
                         set(profile2["contraindications"]))
    shared_effects = list(set(profile1["side_effects"]) &
                          set(profile2["side_effects"]))

    # Check if drug1 treats something drug2 contraindicates and vice versa
    conflict1 = list(set(profile1["treats"]) &
                     set(profile2["contraindications"]))
    conflict2 = list(set(profile2["treats"]) &
                     set(profile1["contraindications"]))

    return {
        "shared_contraindications": shared_contra,
        "shared_side_effects": shared_effects,
        "therapeutic_conflicts": conflict1 + conflict2
    }

# ── STEP 3: LLM SAFETY ASSESSMENT ────────────────────────────────────────────
def assess_safety(drug1: str, drug2: str, profile1: dict,
                  profile2: dict, risks: dict, client=None) -> str:
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)
    """Use LLM to provide a clinical safety assessment."""

    prompt = f"""You are a clinical pharmacology assistant.
Given the following drug profiles from a medical knowledge graph,
assess whether it is safe to take {drug1} and {drug2} together.

{drug1} profile:
- Treats: {', '.join(profile1['treats'][:5]) or 'unknown'}
- Contraindicated for: {', '.join(profile1['contraindications'][:10]) or 'none found'}
- Side effects: {', '.join(profile1['side_effects'][:10]) or 'none found'}

{drug2} profile:
- Treats: {', '.join(profile2['treats'][:5]) or 'unknown'}
- Contraindicated for: {', '.join(profile2['contraindications'][:10]) or 'none found'}
- Side effects: {', '.join(profile2['side_effects'][:10]) or 'none found'}

Shared risks found in KG:
- Shared contraindications: {', '.join(risks['shared_contraindications'][:5]) or 'none'}
- Shared side effects: {', '.join(risks['shared_side_effects'][:5]) or 'none'}
- Therapeutic conflicts: {', '.join(risks['therapeutic_conflicts'][:5]) or 'none'}

Provide:
1. A safety verdict: SAFE / CAUTION / AVOID
2. Reasoning based ONLY on the KG data above
3. Key risks to watch for

Keep it concise and clinical.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ── FULL PIPELINE ─────────────────────────────────────────────────────────────
def check_interaction(drug1: str, drug2: str) -> dict:
    """Full drug-drug interaction check pipeline."""
    print(f"\n{'='*60}")
    print(f"💊 Checking interaction: {drug1} + {drug2}")
    print('='*60)

    print("📊 Fetching drug profiles from Neo4j...")
    profile1 = get_drug_profile(drug1)
    profile2 = get_drug_profile(drug2)

    print(f"  {drug1}: {len(profile1['contraindications'])} contraindications, "
          f"{len(profile1['side_effects'])} side effects")
    print(f"  {drug2}: {len(profile2['contraindications'])} contraindications, "
          f"{len(profile2['side_effects'])} side effects")

    print("🔍 Finding shared risks...")
    risks = find_shared_risks(profile1, profile2)
    print(f"  Shared contraindications: {len(risks['shared_contraindications'])}")
    print(f"  Shared side effects: {len(risks['shared_side_effects'])}")
    print(f"  Therapeutic conflicts: {len(risks['therapeutic_conflicts'])}")

    print("🤖 Running LLM safety assessment...")
    assessment = assess_safety(drug1, drug2, profile1, profile2, risks)

    print(f"\n📋 Safety Assessment:\n{assessment}")

    return {
        "drug1": drug1,
        "drug2": drug2,
        "profile1": profile1,
        "profile2": profile2,
        "risks": risks,
        "assessment": assessment
    }

# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test pairs
    check_interaction("Metformin", "Codeine")
    check_interaction("Lisinopril", "Metformin")