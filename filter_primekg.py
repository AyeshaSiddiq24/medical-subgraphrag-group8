import pandas as pd

# ← update this path to where your kg.csv is
df = pd.read_csv("/Users/ayeshasddiq/Desktop/nexus-medical-subgraphrag/data/raw/kg.csv")


# Keep only medically relevant relations
keep = ["indication", "contraindication", "drug_effect", "disease_disease"]
filtered = df[df["relation"].isin(keep)]

print("Filtered rows:", len(filtered))
print(filtered["relation"].value_counts())

# Save to a new CSV
filtered.to_csv("medical_kg_filtered.csv", index=False)
print("✅ Saved to medical_kg_filtered.csv")