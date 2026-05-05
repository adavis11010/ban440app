"app.py"
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
st.set_page_config(page_title="CourseCast Model Predictor", layout="centered")

RAW_MODEL_URL = "https://raw.githubusercontent.com/adavis11010/ban440app/main/my_model.joblib"

@st.cache_resource
def load_model():
    response = requests.get(RAW_MODEL_URL)
    response.raise_for_status()
    model_bytes = io.BytesIO(response.content)
    model = joblib.load(model_bytes)
    return model
        return joblib.load("coursecast_model.pkl")

model = load_model()

FEATURES = [
    "Previous Scores",
    "Attendance Rate",
    "Hours Studied",
    "Extracurricular Activities"
]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )
    return df[FEATURES]   # drops all other columns automatically

st.title("CourseCast")
st.markdown("Upload a student CSV to flag who is **At-Risk** of failing.")
# Inputs
uploaded_file = st.file_uploader(
    label="Drop your student CSV here",
    type=["csv"],
    help="Must include: Previous Scores, Attendance Rate, Hours Studied, Extracurricular Activities"
)

if uploaded_file is not None:

    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    missing = [col for col in FEATURES if col not in df_raw.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    try:
        X = preprocess(df_raw)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    df_results = df_raw.copy()
    df_results["Risk Label"] = np.where(predictions == 1, "At-Risk", "Not At-Risk")
    df_results["Confidence"] = (probabilities * 100).round(1).astype(str) + "%"

    st.success(f"Processed {len(df_results)} students.")

    col1, col2 = st.columns(2)
    col1.metric("At-Risk", int(predictions.sum()))
    col2.metric("Not At-Risk", int((predictions == 0).sum()))

    st.dataframe(
        df_results[["Previous Scores", "Attendance Rate", "Hours Studied",
                    "Extracurricular Activities", "Risk Label", "Confidence"]],
        use_container_width=True
    )

    st.download_button(
        label="Download Results CSV",
        data=df_results.to_csv(index=False).encode("utf-8"),
        file_name="coursecast_predictions.csv",
        mime="text/csv"
    )
    git init
git add .
git commit -m "initial deploy"
git remote add origin https://github.com/adavis11010/ban440app.git
git push -u origin main
