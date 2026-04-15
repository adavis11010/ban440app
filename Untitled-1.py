"app.py"
import streamlit as st
import joblib
import numpy as np
import requests
import io

st.set_page_config(page_title="Cloud Model Predictor", layout="centered")

RAW_MODEL_URL = "https://raw.githubusercontent.com/adavis11010/ban440app/main/my_model.joblib"

@st.cache_resource
def load_model():
    response = requests.get(RAW_MODEL_URL)
    response.raise_for_status()
    model_bytes = io.BytesIO(response.content)
    model = joblib.load(model_bytes)
    return model

model = load_model()

st.title("📈 Cloud‑Hosted Model Prediction App")
st.write("Enter values for the four features to generate a predicted percent.")

# Inputs
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)

# Predict
if st.button("Predict"):
    X = np.array([[f1, f2, f3, f4]])
    pred = model.predict(X)[0]
    percent = float(pred) * 100

    st.subheader("🔮 Prediction Result")
    st.metric("Predicted Percent", f"{percent:.2f}%")
