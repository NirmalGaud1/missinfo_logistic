import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_model():
    with open("logistic_model_missinfo.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer_missinfo.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

st.title("🚨 Misinformation Detection Using Logistic Regression")
st.write("Enter a tweet below to classify it as Misinformation or Not.")

user_input = st.text_area("Tweet Text", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        X_input = vectorizer.transform([user_input])
        
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][prediction]

        label = "✅ Not Misinformation" if prediction == 0 else "❌ Misinformation"
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** {proba*100:.2f}%")
