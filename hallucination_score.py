"""
Hallucination Score
Measures what % of LLM answers are grounded in retrieved triples vs made up.
Directly replicates the Scoreh metric from the SubgraphRAG paper.

Score per answer:
  - Every answer entity found in retrieved triples → +1 (grounded)
  - Every answer entity NOT found in retrieved triples → -1 (hallucinated)
  - Missing answers → 0

Final score normalized 0-100. Higher = less hallucination.
"""

import json
from pipeline import retrieve_subgraph_and_answer

# ── LOAD RESULTS ──────────────────────────────────────────────────────────────
with open("evaluation_results.json") as f:
    subgraph_results = json.load(f)

with open("baseline_results.json") as f:
    baseline_results = json.load(f)

# ── PARSE ANS LINES ───────────────────────────────────────────────────────────
def parse_answers(text: str) -> list:
    """Extract ans: lines from LLM response."""
    answers = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("ans:"):
            ans = line[4:].strip()
            if ans and ans.lower() != "not available":
                answers.append(ans.lower())
    return answers

# ── CHECK IF ANSWER IS GROUNDED ───────────────────────────────────────────────
def is_grounded(answer: str, triples: list) -> bool:
    """Check if an answer entity appears in the retrieved triples."""
    triple_text = " ".join([t["text"].lower() for t in triples])
    # Extract meaningful words from answer (skip short/common words)
    stopwords = {"the", "and", "for", "with", "not", "are", "was", "ans", "available"}
    answer_words = [w.strip(".,;:()") for w in answer.lower().split()
                    if len(w) > 3 and w not in stopwords]
    if not answer_words:
        return False
    # Answer is grounded if ANY key word appears in triples
    matches = sum(1 for word in answer_words if word in triple_text)
    return matches >= 1

# ── SCORE ONE RESPONSE ────────────────────────────────────────────────────────
def score_response(llm_response: str, triples: list) -> dict:
    """Score a single LLM response for hallucination."""
    answers = parse_answers(llm_response)

    if not answers:
        return {"score": 0, "grounded": 0, "hallucinated": 0, "total": 0}

    grounded    = sum(1 for a in answers if is_grounded(a, triples))
    hallucinated = len(answers) - grounded

    # Score: grounded = +1, hallucinated = -1, normalized 0-100
    raw = (grounded - hallucinated) / len(answers)
    normalized = max(0, (raw + 1) / 2 * 100)

    return {
        "score": round(normalized, 1),
        "grounded": grounded,
        "hallucinated": hallucinated,
        "total": len(answers)
    }

# ── EVALUATE ALL QUESTIONS ────────────────────────────────────────────────────
def evaluate_hallucination(results: list, label: str) -> float:
    """Evaluate hallucination score across all questions."""
    from pipeline import get_triples, extract_entities
    from groq import Groq
    import time

    client = Groq(api_key="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")

    print(f"\n{'='*65}")
    print(f"Hallucination Evaluation — {label}")
    print(f"{'='*65}")
    print(f"{'Question':<45} {'Score':>6} {'Grd':>5} {'Hall':>5}")
    print(f"{'-'*65}")

    total_score = 0
    for item in results:
        query  = item["question"]
        answer = item["answer"]

        # Get triples fresh from Neo4j
        try:
            entities = extract_entities(query, client)
            triples  = []
            for e in entities:
                triples.extend(get_triples(e))
            time.sleep(3)  # avoid rate limit
        except Exception as ex:
            triples = []

        scored = score_response(answer, triples)
        total_score += scored["score"]

        print(f"{query[:44]:<45} {scored['score']:>5.1f} {scored['grounded']:>5} {scored['hallucinated']:>5}")

    avg = total_score / len(results)
    print(f"{'-'*65}")
    print(f"{'Average Hallucination Score':<45} {avg:>5.1f}")
    return avg

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 Evaluating Hallucination Scores...")
    print("Higher score = less hallucination = better\n")

    sub_score  = evaluate_hallucination(subgraph_results, "SubgraphRAG")
    base_score = evaluate_hallucination(baseline_results, "Baseline RAG")

    print(f"\n{'='*65}")
    print(f"FINAL COMPARISON")
    print(f"{'='*65}")
    print(f"SubgraphRAG Hallucination Score : {sub_score:.1f}/100")
    print(f"Baseline RAG Hallucination Score: {base_score:.1f}/100")
    print(f"Improvement                     : +{sub_score - base_score:.1f} points")
    print(f"{'='*65}")