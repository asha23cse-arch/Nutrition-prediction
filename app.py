
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import softmax

st.set_page_config(page_title="BDHS ML Prediction", layout="wide")
st.title("Bangladesh DHS 2022")
st.subheader("Education Level Prediction")

@st.cache_resource
def load_models():
    model = joblib.load("deployment/xgb_tuned.joblib")
    imputer = joblib.load("deployment/imputer.joblib")
    scaler = joblib.load("deployment/scaler.joblib")
    with open("deployment/config.json", encoding="utf-8") as f:
        config = json.load(f)
    return model, imputer, scaler, config

try:
    model, imputer, scaler, config = load_models()
except Exception as e:
    st.error("Could not load model files.")
    st.exception(e)
    st.stop()

features = list(imputer.feature_names_in_)
selected_features = config["features"]
class_names = config["class_names"]
temperature = float(config.get("temperature", 1.0))

st.info("Upload a CSV with the required encoded input features.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = [f for f in features if f not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        try:
            X = df[features].apply(pd.to_numeric, errors="coerce")
            X = imputer.transform(X)
            X = scaler.transform(X)
            X = pd.DataFrame(X, columns=features)
            X = X[selected_features]

            p = softmax(
                np.log(model.predict_proba(X) + 1e-12)
                / max(temperature, 1e-8),
                axis=1
            )

            idx = np.argmax(p, axis=1)
            result = pd.DataFrame({
                "Predicted Education Level": [
                    class_names[i] for i in idx
                ],
                "Confidence": np.max(p, axis=1)
            })

            st.subheader("Prediction Results")
            st.dataframe(result, use_container_width=True)
            st.bar_chart(result["Confidence"])

            st.download_button(
                "Download Predictions",
                result.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error("Prediction failed. Check preprocessing compatibility.")
            st.exception(e)
