from neo4j import GraphDatabase

URI      = "neo4j://127.0.0.1:7687"
USER     = "neo4j"
PASSWORD = "password123"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def get_triples(entity):
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(a.name) CONTAINS toLower($entity)
               OR toLower(b.name) CONTAINS toLower($entity)
            RETURN a.name AS head, r.type AS relation, b.name AS tail
            LIMIT 50
        """, entity=entity)
        triples = []
        for record in result:
            triples.append({
                "head": record["head"],
                "relation": record["relation"],
                "tail": record["tail"],
                "text": f"({record['head']}, {record['relation']}, {record['tail']})"
            })
        return triples

# Test it
entities = ["Alzheimer disease"]
for entity in entities:
    triples = get_triples(entity)
    print(f"\nTriples for '{entity}':")
    for t in triples[:10]:
        print(t["text"])
    print(f"Total: {len(triples)} triples found")

driver.close()