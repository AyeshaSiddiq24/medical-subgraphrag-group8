from pipeline import run

print("\n" + "="*60)
print("🏥  Medical SubgraphRAG Assistant")
print("="*60)
print("Ask any medical question about diseases, drugs,")
print("side effects, or contraindications.")
print("Type 'quit' to exit.\n")

while True:
    query = input("You: ").strip()
    
    if query.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    
    if not query:
        continue

    run(query)
    print()