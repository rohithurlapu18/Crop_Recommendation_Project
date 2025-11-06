from neo4j import GraphDatabase

# Connection settings
uri = "bolt://localhost:7687"  # default Neo4j Bolt port
username = "neo4j"             # your Neo4j username
password = "12345678"              # replace with your real Neo4j password

try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        result = session.run("RETURN '✅ Connection Successful!' AS message")
        for record in result:
            print(record["message"])
    driver.close()
except Exception as e:
    print("❌ Connection failed:", e)
