import pandas as pd
from neo4j import GraphDatabase

URI      = "neo4j://127.0.0.1:7687"
USER     = "neo4j"
PASSWORD = "password123"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
df = pd.read_csv("medical_kg_filtered.csv", low_memory=False)

print(f"Loading {len(df)} rows into Neo4j...")

def load_batch(tx, batch):
    for _, row in batch.iterrows():
        tx.run("""
            MERGE (a {name: $head, type: $head_type})
            SET a.type = $head_type
            MERGE (b {name: $tail, type: $tail_type})
            SET b.type = $tail_type
            MERGE (a)-[:RELATION {type: $relation}]->(b)
        """, head=row["x_name"], head_type=row["x_type"],
             tail=row["y_name"], tail_type=row["y_type"],
             relation=row["relation"])

BATCH_SIZE = 500
total = len(df)

for i in range(0, total, BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]
    with driver.session(database="neo4j") as session:
        session.execute_write(load_batch, batch)
    print(f"  Loaded {min(i+BATCH_SIZE, total)}/{total} rows...")

driver.close()
print("✅ Done loading data into Neo4j!")