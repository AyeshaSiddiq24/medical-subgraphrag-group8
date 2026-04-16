
from groq import Groq
import json

client = Groq(api_key="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")  # ← paste your gsk_ key

def extract_entities(query):
    prompt = f"""
Extract all medical entities from this question.
Return ONLY a JSON list. Example: ["Hypertension", "Lisinopril"]

Question: {query}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# Test it
query = "What drugs are used to treat Alzheimer's disease?"
entities = extract_entities(query)
print("Extracted entities:", entities)