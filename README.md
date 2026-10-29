"""Streamlit interface for the insurance cost predictor."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/best_model.joblib")
META_PATH = Path("artifacts/model_metadata.json")

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Insurance Cost Predictor")
st.write("Estimate medical insurance charges using a machine-learning model.")
st.caption("Educational estimate only—not an official insurance quote or medical/financial advice.")

if not MODEL_PATH.exists():
    st.error("The trained model was not found.")
    st.markdown("Run `python download_dataset.py` and then `python train_model.py` before starting the app.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}

with st.form("prediction_form"):
    st.subheader("Enter your details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        sex = st.selectbox("Sex", ["female", "male"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.5, step=0.1)
    with col2:
        children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southeast", "southwest", "northwest", "northeast"])
    submitted = st.form_submit_button("Predict insurance charges", type="primary")

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])
    prediction = float(model.predict(input_df)[0])
    st.subheader("Estimated annual medical charges")
    st.metric("Predicted charges", f"${prediction:,.2f}")
    st.caption(f"Model selected by 5-fold CV RMSE: {metadata.get('best_model', 'Best model')}")
    st.info("Actual charges can differ substantially. The model reflects patterns in its training dataset and may not generalize to every person.")

with st.expander("About this model"):
    st.write("Models compared: Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.")
    st.write("Metrics: MAE, RMSE, and R², averaged across 5 cross-validation folds.")
    st.write("Categorical fields are one-hot encoded. Numeric fields are median-imputed and standardized inside the training pipeline.")
