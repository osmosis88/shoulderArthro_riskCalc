import streamlit as st
import pickle
import pandas as pd

# 🎨 Custom Styling
st.set_page_config(page_title="A Risk Calculator for Postoperative Shoulder Arthroplasty Complications", layout="wide")

# 🎯 Define model file paths
model_paths = {
    "Serious Complication 🤕": "gbc_for_seriousComp.pk1",
    "Medical Complication 🏥": "gbc_for_medicalComp.pk1",
    "Surgical Complication 🏷️": "gbc_for_surgicalComp.pkl",
    "Any Complication ⚠️": "gbc_for_anyComp.pkl"
}

# 🚀 Load the trained models
@st.cache_resource
def load_models():
    models = {}
    for name, path in model_paths.items():
        with open(path, "rb") as file:
            models[name] = pickle.load(file)
    return models

models = load_models()

# 🏥 Title of the Streamlit App
st.markdown("<h1>Shoulder Arthroplasty Postoperative Risk Calculator</h1>", unsafe_allow_html=True)

st.markdown("""
### 🏥 **Enter Patient Information**
Fill out the form below to get the predicted probabilities for:
- **Serious Complications 🤕**
- **Medical Complications 🏥**
- **Surgical Complications 🏷️**
- **Any Complication ⚠️**
""")

# ✍️ User input form
with st.form("prediction_form"):
    st.subheader("📝 Patient Data Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        DM = st.selectbox("🩸 Diabetes (DM)", ["No", "Yes"])
        COPD = st.selectbox("💨 Chronic Obstructive Pulmonary Disease (COPD)", ["No", "Yes"])
        HTN = st.selectbox("💔 Hypertension (HTN)", ["No", "Yes"])
        bleedingDisorder = st.selectbox("🩸 Bleeding Disorder", ["No", "Yes"])
    
    with col2:
        smokingStatus = st.selectbox("🚬 Smoking Status", ["No", "Yes"])
        FHS = st.selectbox("🏃 Functional Health Status", ["Independent", "Dependent"])
        ASA = st.selectbox("📊 ASA Classification", ["ASA I", "ASA II", "ASA III", "ASA IV", "ASA V"])

    with col3:
        Albumin = st.number_input("🧪 Albumin Level", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        HCT = st.number_input("🩸 Hematocrit (HCT) Level", min_value=0.0, max_value=100.0, value=42.0, step=0.1)
        BUN = st.number_input("🧫 Blood Urea Nitrogen (BUN) Level", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    
    st.markdown("---")

    # 🔘 Submit button
    submit_button = st.form_submit_button(label="🔍 Predict")

# 📊 Mapping categorical inputs to binary values
categorical_mappings = {
    "No": 0, "Yes": 1,
    "Independent": 0, "Dependent": 1
}

# ✅ ASA Mapping: Binary for ASA I & II (0) vs. ASA III, IV, V (1)
asa_mapped_value = 1 if ASA in ["ASA III", "ASA IV", "ASA V"] else 0

# ✅ Albumin Mapping: Low Albumin (<3.5) is 1, Normal (≥3.5) is 0
albumin_mapped_value = 1 if Albumin < 3.5 else 0

# ✅ HCT Mapping: Low HCT (<39) is 1, Normal (≥39) is 0
hct_mapped_value = 1 if HCT < 39 else 0

# ✅ BUN Mapping: High BUN (>20) is 1, Normal (≤20) is 0
bun_mapped_value = 1 if BUN > 20 else 0

# 🎯 If user submits the form, make predictions
if submit_button:
    input_data = pd.DataFrame([[
        categorical_mappings[DM], categorical_mappings[smokingStatus], categorical_mappings[FHS],
        categorical_mappings[COPD], categorical_mappings[HTN], categorical_mappings[bleedingDisorder],
        albumin_mapped_value, hct_mapped_value, bun_mapped_value, asa_mapped_value
    ]], columns=[
        "DM_Yes", "smokingStatus_Yes", "FHS_Yes", "COPD_Yes", "HTN_Yes", "bleedingDisorder_Yes", 
        "Albumin_Normal", "HCT_Normal", "BUN_Normal", "ASA_ASA_II"
    ])

    # 📊 Store predictions
    predictions = {}
    
    for model_name, model in models.items():
        prob = model.predict_proba(input_data)[0][1]
        predictions[model_name] = round(prob, 4)

    # 🔥 Display results
    st.subheader("📊 Prediction Probabilities")
    for comp, prob in predictions.items():
        st.markdown(f"**{comp}:** <span style='font-size:22px; font-weight:bold; color:#37474F;'>{prob * 100:.2f}%</span>", unsafe_allow_html=True)
