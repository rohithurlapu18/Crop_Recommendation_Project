# ======================================================
# Hybrid Crop Recommendation (ML + Knowledge Graph)
# ======================================================

import numpy as np
import joblib
from neo4j import GraphDatabase

# ---------- 1. Load trained ML model ----------
ml_model = joblib.load("models/xgboost_crop_model.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# ---------- 2. Connect to Neo4j ----------
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"   # your password
driver = GraphDatabase.driver(uri, auth=(username, password))

# ---------- 3. Hybrid recommendation function ----------
def hybrid_recommend(input_features, prev_crop=None, season=None):
    # Step A: ML prediction
    probs = ml_model.predict_proba(np.array([input_features]))[0]
    top_indices = probs.argsort()[-5:][::-1]
    top_crops = [label_encoder.inverse_transform([i])[0].lower() for i in top_indices]  # lowercase
    print("\n🌱 ML Top-5 Suggestions:")
    for c in top_crops:
        print("  -", c)

    # Step B: Knowledge Graph reasoning
    print("\n🧠 Applying Knowledge Graph Filters...")

    filtered_crops = []
    with driver.session() as session:
        for crop in top_crops:
            kg_score = 1.0

            # Rule 1: Crop rotation — avoid same as previous crop
            if prev_crop:
                query = """
                    MATCH (a:Crop)-[:ROTATION_WITH]->(b:Crop)
                    WHERE toLower(a.name) = toLower($prev)
                      AND toLower(b.name) = toLower($crop)
                    RETURN count(b) AS rel
                """
                res = session.run(query, prev=prev_crop, crop=crop)
                rel_count = res.single()["rel"]
                if rel_count == 0:
                    kg_score -= 0.2  # penalty
                else:
                    kg_score += 0.1  # bonus

            # Rule 2: Season suitability
            if season:
                query = """
                    MATCH (c:Crop)-[:SUITABLE_FOR]->(s:Season)
                    WHERE toLower(c.name) = toLower($crop)
                      AND toLower(s.name) = toLower($season)
                    RETURN count(s) AS ok
                """
                res = session.run(query, crop=crop, season=season)
                season_match = res.single()["ok"]
                if season_match == 0:
                    kg_score -= 0.3
                else:
                    kg_score += 0.2

            filtered_crops.append((crop, kg_score))

    # Step C: Re-rank
    final_rank = sorted(filtered_crops, key=lambda x: x[1], reverse=True)

    print("\n🌾 Final Hybrid Recommendations:")
    for crop, score in final_rank:
        print(f"  {crop:12s} | Hybrid Score = {score:.2f}")


# ---------- 4. Example test ----------
# Input: [N, P, K, temperature, humidity, pH, rainfall]
input_features = [90, 42, 43, 21.0, 82.0, 6.5, 200]
hybrid_recommend(input_features, prev_crop="Rice", season="Rabi")
driver.close()
