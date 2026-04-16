import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.title("🛒 RecoSphere – Recommendation System")

data = pd.DataFrame({
    "item": ["A", "B", "C", "D"],
    "feature1": [1, 0, 1, 0],
    "feature2": [0, 1, 1, 0]
})

item = st.selectbox("Select Item", data["item"])

if st.button("Recommend"):
    sim = cosine_similarity(data.iloc[:,1:])
    idx = data[data["item"] == item].index[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = [data.iloc[i[0]]["item"] for i in scores[1:3]]
    st.write("Recommended:", recommendations)
