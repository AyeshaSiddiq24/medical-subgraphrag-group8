import time
from pipeline import run
import json

# 10 test questions based on your dataset
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

results = []

for i, q in enumerate(questions):
    print(f"\n🧪 Test {i+1}/10")
    answer = run(q)
    results.append({"question": q, "answer": answer})
    time.sleep(10)  # wait 10 seconds between questions

# Save results to file
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Evaluation done! Results saved to evaluation_results.json")