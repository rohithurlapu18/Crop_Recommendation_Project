# ======================================================
# build_knowledge_graph.py
# Creates Knowledge Graph for Crop Recommendation System
# ======================================================

from neo4j import GraphDatabase

# ---------- Neo4j Connection ----------
uri = "bolt://localhost:7687"
username = "neo4j"       # your Neo4j username
password = "12345678"        # your Neo4j password
driver = GraphDatabase.driver(uri, auth=(username, password))

# ---------- Define Nodes and Relationships ----------

def create_nodes(tx):
    # Crops
    crops = ["rice", "wheat", "maize", "pulses", "groundnut", "cotton", "banana", "jute", "coffee"]
    for crop in crops:
        tx.run("MERGE (:Crop {name: $name})", name=crop)

    # Soils
    soils = ["clay", "loamy", "sandy", "sandy loam"]
    for soil in soils:
        tx.run("MERGE (:Soil {type: $type})", type=soil)

    # Seasons
    seasons = ["Kharif", "Rabi", "Zaid"]
    for season in seasons:
        tx.run("MERGE (:Season {name: $name})", name=season)

def create_relationships(tx):
    relationships = [
        # GROWS_IN relationships
        ("rice", "clay", "GROWS_IN"),
        ("wheat", "loamy", "GROWS_IN"),
        ("maize", "sandy loam", "GROWS_IN"),
        ("pulses", "loamy", "GROWS_IN"),
        ("groundnut", "sandy", "GROWS_IN"),
        ("cotton", "sandy loam", "GROWS_IN"),
        ("banana", "clay", "GROWS_IN"),
        ("jute", "clay", "GROWS_IN"),
        ("coffee", "loamy", "GROWS_IN"),

        # SUITABLE_FOR relationships
        ("rice", "Kharif", "SUITABLE_FOR"),
        ("wheat", "Rabi", "SUITABLE_FOR"),
        ("maize", "Kharif", "SUITABLE_FOR"),
        ("pulses", "Rabi", "SUITABLE_FOR"),
        ("groundnut", "Kharif", "SUITABLE_FOR"),
        ("cotton", "Kharif", "SUITABLE_FOR"),
        ("banana", "Kharif", "SUITABLE_FOR"),
        ("jute", "Kharif", "SUITABLE_FOR"),
        ("coffee", "Zaid", "SUITABLE_FOR"),

        # ROTATION_WITH relationships
        ("rice", "wheat", "ROTATION_WITH"),
        ("maize", "pulses", "ROTATION_WITH"),
        ("groundnut", "cotton", "ROTATION_WITH"),
        ("rice", "pulses", "ROTATION_WITH"),
        ("wheat", "maize", "ROTATION_WITH"),
        ("banana", "coffee", "ROTATION_WITH")
    ]

    for a, b, rel in relationships:
        tx.run(f"""
            MATCH (x {{name:$a}}), (y {{name:$b}}) 
            MERGE (x)-[:{rel}]->(y)
        """, a=a, b=b)

# ---------- Build Graph ----------

def build_graph():
    with driver.session() as session:
        session.execute_write(create_nodes)
        session.execute_write(create_relationships)
    print("✅ Knowledge Graph successfully created in Neo4j!")

# ---------- Run ----------
if __name__ == "__main__":
    try:
        build_graph()
    except Exception as e:
        print("❌ Error building graph:", e)
    finally:
        driver.close()
