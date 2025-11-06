import streamlit as st
import numpy as np
import joblib
from neo4j import GraphDatabase

# ---------- Load model ----------
ml_model = joblib.load("models/xgboost_crop_model.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# ---------- Connect to Neo4j ----------
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))

# ---------- Hybrid recommendation function ----------
def hybrid_recommend(input_features, prev_crop=None, season=None):
    probs = ml_model.predict_proba(np.array([input_features]))[0]
    top_indices = probs.argsort()[-5:][::-1]
    top_crops = [label_encoder.inverse_transform([i])[0].lower() for i in top_indices]

    results = []
    with driver.session() as session:
        for crop in top_crops:
            kg_score = 1.0

            # Rotation logic
            if prev_crop:
                query = """
                    MATCH (a:Crop)-[:ROTATION_WITH]->(b:Crop)
                    WHERE toLower(a.name) = toLower($prev)
                      AND toLower(b.name) = toLower($crop)
                    RETURN count(b) AS rel
                """
                rel_count = session.run(query, prev=prev_crop, crop=crop).single()["rel"]
                if rel_count == 0:
                    kg_score -= 0.2
                else:
                    kg_score += 0.1

            # Season logic
            if season:
                query = """
                    MATCH (c:Crop)-[:SUITABLE_FOR]->(s:Season)
                    WHERE toLower(c.name) = toLower($crop)
                      AND toLower(s.name) = toLower($season)
                    RETURN count(s) AS ok
                """
                season_match = session.run(query, crop=crop, season=season).single()["ok"]
                if season_match == 0:
                    kg_score -= 0.3
                else:
                    kg_score += 0.2

            results.append((crop, round(kg_score, 2)))
    return results

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Hybrid Crop Recommendation", layout="wide")
st.title("🌾 Hybrid ML + Knowledge Graph Crop Recommendation System")

st.write("### Enter Soil and Environmental Parameters:")

col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=140, value=42)
    K = st.number_input("Potassium (K)", min_value=0, max_value=140, value=43)
with col2:
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
with col3:
    ph = st.number_input("pH", min_value=3.0, max_value=10.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=200.0)

prev_crop = st.text_input("Previous Crop (optional)", "rice")
season = st.selectbox("Current Season", ["kharif", "rabi", "zaid"])

if st.button("🔍 Recommend Crops"):
    input_features = [N, P, K, temp, humidity, ph, rainfall]
    recommendations = hybrid_recommend(input_features, prev_crop, season)

    st.subheader("🌾 Final Hybrid Recommendations")
    st.table(recommendations)

driver.close()
