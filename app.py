import streamlit as st
import joblib 
import pandas as pd
import numpy as np

# 🎨 Custom Styling
st.set_page_config(page_title="A Risk Calculator for Postoperative Shoulder Arthroplasty Complications", layout="wide")

# Custom CSS to enhance UI
st.markdown("""
    <style>
        .stButton>button {
            color: white !important;
            background-color: #0288D1 !important;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #0277BD !important;
        }
        .big-font {
            font-size: 22px !important;
            font-weight: bold;
            color: #37474F;
        }
        .title-font {
            font-size: 28px !important;
            font-weight: bold;
            color: #1E88E5;
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Define model file paths (Updated extensions from .pkl to .joblib)
model_paths = {
    "Serious Complication 🤕": "gradient_boosting_model_seriousComp.joblib",
    "Medical Complication 🏥": "gradient_boosting_model_medicalComp.joblib",
    "Surgical Complication 🏷️": "gradient_boosting_model_surgicalComp.joblib"
}

# 🚀 Load the trained models using joblib
@st.cache_resource
def load_models():
    models = {}
    for name, path in model_paths.items():
        models[name] = joblib.load(path)  # ✅ Load using joblib
    return models

models = load_models()

# 🏥 Title of the Streamlit App
st.markdown('<p class="title-font">Shoulder Arthroplasty Postoperative Risk Calculator</p>', unsafe_allow_html=True)

st.markdown("""
### 🏥 **Enter Patient Information**
Fill out the form below to get the predicted probabilities for:
- **Serious Complications 🤕**
- **Medical Complications 🏥**
- **Surgical Complications 🏷️**
""")

# ✍️ User input form
with st.form("prediction_form"):
    st.subheader("📝 Patient Data Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        DM = st.selectbox("🩸 Diabetes (DM)", ["No", "Yes"])
        COPD = st.selectbox("💨 Chronic Obstructive Pulmonary Disease (COPD)", ["No", "Yes"])
        preopTransfusion = st.selectbox("💉 Preoperative Transfusion", ["No", "Yes"])

    with col2:
        smokingStatus = st.selectbox("🚬 Smoking Status", ["No", "Yes"])
        HTN = st.selectbox("💔 Hypertension (HTN)", ["No", "Yes"])
        bleedingDisorder = st.selectbox("🩸 Bleeding Disorder", ["No", "Yes"])

    with col3:
        FHS = st.selectbox("🏃 Functional Health Status", ["Independent", "Dependent"])
        ASA = st.selectbox("📊 ASA Classification", ["ASA I", "ASA II", "ASA III", "ASA IV", "ASA V"])

    st.markdown("---")

    col4, col5, col6 = st.columns(3)

    with col4:
        Albumin = st.number_input("🧪 Albumin Level", min_value=0.0, max_value=30.0, value=3.5, step=0.1)

    with col5:
        HCT = st.number_input("🩸 Hematocrit (HCT) Level", min_value=0.0, max_value=100.0, value=42.0, step=0.1)

    with col6:
        BUN = st.number_input("🧫 Blood Urea Nitrogen (BUN) Level", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    st.markdown("---")

    # 🔘 Submit button
    submit_button = st.form_submit_button(label="🔍 Predict")

# 📊 Mapping categorical inputs to numerical values
categorical_mappings = {
    "No": 0, "Yes": 1,
    "Independent": 0, "Dependent": 1
}

# ✅ Correct ASA Mapping:
asa_mapped_value = 0 if ASA in ["ASA I", "ASA II"] else 1

# ✅ Correct Albumin Mapping:
if Albumin < 3.5:
    albumin_mapped_value = 1
elif 3.5 <= Albumin <= 5.5:
    albumin_mapped_value = 2
else:
    albumin_mapped_value = 3

# ✅ Correct HCT Mapping:
if HCT < 39:
    hct_mapped_value = 1
elif 39 <= HCT <= 49:
    hct_mapped_value = 2
else:
    hct_mapped_value = 3

# ✅ Correct BUN Mapping:
if BUN < 5:
    bun_mapped_value = 1
elif 5 <= BUN <= 20:
    bun_mapped_value = 2
else:
    bun_mapped_value = 3

# 🎯 If user submits the form, make predictions
if submit_button:
    input_data = pd.DataFrame([[
        categorical_mappings[DM],
        categorical_mappings[smokingStatus],
        categorical_mappings[FHS],
        categorical_mappings[COPD],
        categorical_mappings[HTN],
        categorical_mappings[bleedingDisorder],
        categorical_mappings[preopTransfusion],
        albumin_mapped_value, hct_mapped_value, bun_mapped_value,
        asa_mapped_value
    ]], columns=[
        "DM", "smokingStatus", "FHS", "COPD", "HTN", "bleedingDisorder", "preopTransfusion",
        "Albumin", "HCT", "BUN", "ASA"
    ])

    # 📊 Store predictions
    predictions = {}
    
    for model_name, model in models.items():
        prob = model.predict_proba(input_data)[0][1]
        predictions[model_name] = round(prob, 4)

    # 🔥 Display results
    st.subheader("📊 Prediction Probabilities")
    for comp, prob in predictions.items():
        st.markdown(f"**{comp}:** <span class='big-font'>{prob * 100:.2f}%</span>", unsafe_allow_html=True)
