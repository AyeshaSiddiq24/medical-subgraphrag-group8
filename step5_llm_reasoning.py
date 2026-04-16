from groq import Groq

client = Groq(api_key="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")  # ← paste your gsk_ key

def ask_llm(query, triples):
    triple_text = "\n".join([t["text"] for t in triples])

    prompt = f"""You are a medical assistant. Use ONLY the triples below to answer the question.
Reason step by step. End with answers prefixed with "ans:".
If the answer is not in the triples, say "ans: not available".

Triples:
{triple_text}

Question: {query}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Test it with mock triples
triples = [
    {"text": "(Donepezil, indication, Alzheimer disease)"},
    {"text": "(Memantine, indication, Alzheimer disease)"},
    {"text": "(Donepezil, side_effect, Nausea)"},
    {"text": "(Donepezil, side_effect, Diarrhea)"},
]

query = "What drugs treat Alzheimer disease and what are their side effects?"
answer = ask_llm(query, triples)
print(answer)