from neo4j import GraphDatabase

URI      = "neo4j://127.0.0.1:7687"
USER     = "neo4j"
PASSWORD = "password123"  # ← change this

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

with driver.session(database="neo4j") as session:
    result = session.run("MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 5")
    for record in result:
        print(record["a.name"], "-->", record["type(r)"], "-->", record["b.name"])

driver.close()