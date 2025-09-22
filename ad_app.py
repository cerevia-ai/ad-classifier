import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version warnings (optional)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# -----------------------------
# 1. Load Model & Preprocessor
# -----------------------------
@st.cache_resource
def load_model_and_preprocessor():
    try:
        with st.spinner("🧠 Loading XGBoost model..."):
            model = joblib.load("XGBoost.joblib")
        with st.spinner("⚙️ Loading preprocessor..."):
            preprocessor = joblib.load("scaler_encoder.pkl")
        return model, preprocessor
    except Exception as e:
        st.error(f"🚨 Error loading model or preprocessor:\n\n`{e}`")
        st.stop()

# Load model — this will show spinners and messages
model, preprocessor = load_model_and_preprocessor()

# -----------------------------
# 2. Show Main UI Only After Load
# -----------------------------
st.title("🧠 Cognitive Status Classification (CN, MCI, AD)")
st.markdown("""
This tool predicts cognitive status using clinical and cognitive data.
Enter patient features below to get a prediction with confidence and explanation.
""")

# -----------------------------
# 3. User Input
# -----------------------------
st.sidebar.header("📊 Patient Features")

def user_input_features():
    age = st.sidebar.slider("Age", 50, 95, 75)
    education = st.sidebar.slider("Years of Education", 6, 20, 16)
    moca = st.sidebar.slider("MoCA Score", 0, 30, 26)
    adas13 = st.sidebar.slider("ADAS13 Score", 0.0, 70.0, 10.0)
    cdsum = st.sidebar.slider("CDR Sum of Boxes", 0.0, 18.0, 1.0)
    faq = st.sidebar.slider("FAQ Total", 0, 30, 5)

    gender = st.sidebar.radio("Gender", ["Female", "Male"])
    ptgender = 2 if gender == "Female" else 1

    ethnicity = st.sidebar.selectbox(
        "Ethnicity (PTETHCAT)",
        [1, 2],
        index=1,
        format_func=lambda x: "Hispanic" if x == 1 else "Not Hispanic"
    )

    race = st.sidebar.selectbox(
        "Race (PTRACCAT)",
        [1, 2, 3, 4, 5, 6, 7],
        format_func=lambda x: {
            1: "Caucasian",
            2: "African American",
            3: "Asian",
            4: "Pacific Islander",
            5: "American Indian",
            6: "More than one race",
            7: "Unknown"
        }[x]
    )

    data = {
        "AGE": [age],
        "ADAS13": [adas13],
        "CDSUM": [cdsum],
        "FAQTOTAL": [faq],
        "MOCA": [moca],
        "PTEDUCAT": [education],
        "PTGENDER": [ptgender],
        "PTETHCAT": [ethnicity],
        "PTRACCAT": [race]
    }
    return pd.DataFrame(data)

input_df = user_input_features()

# -----------------------------
# 4. Preprocess Input
# -----------------------------
try:
    X_processed = preprocessor.transform(input_df)
except Exception as e:
    st.error("🚨 Error during preprocessing. Check input or preprocessor compatibility.")
    st.stop()

# -----------------------------
# 5. Predict
# -----------------------------
y_proba = model.predict_proba(X_processed)
y_pred = model.predict(X_processed)[0]

class_names = ['CN', 'MCI', 'AD']
predicted_label = class_names[y_pred]
confidence = np.max(y_proba, axis=1)[0]

# Format probabilities
proba_dict = {cls: prob for cls, prob in zip(class_names, y_proba[0])}

st.subheader("✅ Prediction Result")
st.write(f"**Predicted Cognitive Status:** `{predicted_label}`")
st.write(f"**Confidence:** `{confidence:.2%}`")

st.write("### 📊 Prediction Probabilities")
cols = st.columns(3)
for i, (cls, prob) in enumerate(proba_dict.items()):
    cols[i].metric(label=cls, value=f"{prob:.2%}")

def add_footer():
    st.markdown("---")
    st.caption("For research and planning use only. Not a medical device.")

add_footer()
