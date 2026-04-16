import json

with open("evaluation_results.json") as f:
    subgraph = json.load(f)

with open("baseline_results.json") as f:
    baseline = json.load(f)

print("\n" + "="*70)
print("COMPARISON: SubgraphRAG vs Baseline RAG")
print("="*70)

subgraph_score = 0
baseline_score = 0

for i, (s, b) in enumerate(zip(subgraph, baseline)):
    s_available = "not available" not in s["answer"].lower()
    b_available = "not available" not in b["answer"].lower()

    if s_available:
        subgraph_score += 1
    if b_available:
        baseline_score += 1

    print(f"\nQ{i+1}: {s['question']}")
    print(f"  SubgraphRAG : {'✅ Answered' if s_available else '❌ Not available'}")
    print(f"  Baseline    : {'✅ Answered' if b_available else '❌ Not available'}")

print("\n" + "="*70)
print(f"SubgraphRAG Score : {subgraph_score}/10")
print(f"Baseline Score    : {baseline_score}/10")
print(f"Improvement       : +{subgraph_score - baseline_score} questions answered")
print("="*70)