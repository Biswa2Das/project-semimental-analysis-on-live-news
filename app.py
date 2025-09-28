# app.py (scikit-learn fallback, Streamlit Community Cloud-friendly)
import streamlit as st
import joblib
import os

@st.cache_resource
def load_model():
    pkl_path = os.getenv("SK_MODEL_PATH", "news_sentiment_sklearn.pkl")
    return joblib.load(pkl_path)

model = load_model()

st.title("News Sentiment (scikit-learn TF-IDF + LR)")
headline = st.text_input("Enter a news headline")
if st.button("Predict") and headline:
    pred = model.predict([headline])[0]
    label = "Positive" if int(pred) == 1 else "Negative"
    st.success(f"Prediction: {label}")
