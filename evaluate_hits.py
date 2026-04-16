import json

# Ground truth answers for our 10 test questions
ground_truth = {
    "What drugs treat Alzheimer disease?": 
        ["Galantamine", "Donepezil", "Rivastigmine", "Memantine", "Tacrine"],
    "What are the contraindications of Codeine?": 
        ["liver disease", "asthma", "epilepsy", "hypertension", "kidney disease"],
    "What drugs treat Parkinson disease?": 
        ["Levodopa", "Carbidopa", "Pramipexole", "Ropinirole", "Selegiline"],
    "What are the side effects of Galantamine?": 
        ["Nausea", "Vomiting", "Diarrhea", "Bradycardia", "Fatigue"],
    "What drugs treat hypertension?": 
        ["Lisinopril", "Amlodipine", "Metoprolol", "Losartan", "Atenolol"],
    "What are the contraindications of Metformin?": 
        ["kidney disease", "liver disease", "lactic acidosis", "heart failure"],
    "What drugs treat epilepsy?": 
        ["Valproic acid", "Carbamazepine", "Phenytoin", "Lamotrigine", "Diazepam"],
    "What are the side effects of Donepezil?": 
        ["Nausea", "Vomiting", "Headache", "Fatigue", "Bradycardia"],
    "What drugs treat asthma?": 
        ["Albuterol", "Budesonide", "Montelukast", "Theophylline", "Salmeterol"],
    "What are the contraindications of Aspirin?": 
        ["asthma", "bleeding disorder", "peptic ulcer"]
}

def hit_at_1(predicted_answer: str, true_answers: list) -> int:
    """Check if any ground truth answer appears in the predicted answer."""
    predicted_lower = predicted_answer.lower()
    for truth in true_answers:
        if truth.lower() in predicted_lower:
            return 1
    return 0

def parse_ans_lines(answer: str) -> str:
    """Extract lines with 'ans:' prefix."""
    lines = [l.strip() for l in answer.splitlines() if l.lower().startswith("ans:")]
    return " ".join(lines) if lines else answer

# ── EVALUATE SUBGRAPHRAG ──────────────────────────────────────────────────────
with open("evaluation_results.json") as f:
    subgraph_results = json.load(f)

with open("baseline_results.json") as f:
    baseline_results = json.load(f)

print("=" * 65)
print(f"{'Question':<45} {'SubRAG':>8} {'Baseline':>8}")
print("=" * 65)

subgraph_hits = 0
baseline_hits = 0

for s, b in zip(subgraph_results, baseline_results):
    q = s["question"]
    truths = ground_truth.get(q, [])

    s_ans = parse_ans_lines(s["answer"])
    b_ans = parse_ans_lines(b["answer"])

    s_hit = hit_at_1(s_ans, truths)
    b_hit = hit_at_1(b_ans, truths)

    subgraph_hits += s_hit
    baseline_hits += b_hit

    s_label = "✅ Hit" if s_hit else "❌ Miss"
    b_label = "✅ Hit" if b_hit else "❌ Miss"
    print(f"{q[:44]:<45} {s_label:>8} {b_label:>8}")

print("=" * 65)
print(f"{'Hit@1 Score':<45} {subgraph_hits}/10    {baseline_hits}/10")
print(f"{'Hit@1 Rate':<45} {subgraph_hits/10*100:.0f}%       {baseline_hits/10*100:.0f}%")
print("=" * 65)