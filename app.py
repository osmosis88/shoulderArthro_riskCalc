import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Shoulder Arthroplasty APEX-HBD SCORe Risk Prediction", layout="centered")
st.markdown("<h2 style='text-align: center;'>🩺 Shoulder Arthroplasty APEX-HBD SCORe Risk Prediction</h2>", unsafe_allow_html=True)
st.markdown("---")

# Select complication type
complication_type = st.selectbox(
    "Select a complication type to predict",
    ["medicalComp", "surgicalComp", "any_comp", "seriousComp"]
)

# Load model and encoders
model, label_encoders = joblib.load(f"gbm_{complication_type}.joblib")
input_vars = list(label_encoders.keys())

# Friendly display name map
display_names = {
    "DM": "Diabetes",
    "HTN": "Hypertension",
    "COPD": "COPD",
    "FHS": "Functional Health Status",
    "smokingStatus": "Smoking Status",
    "preopTransfusion": "Transfusion",
    "bleedingDisorder": "Bleeding Disorder",
    "Albumin": "Albumin (g/dL)",
    "HCT": "Hematocrit (%)",
    "BUN": "BUN (mg/dL)",
    "ASA": "ASA Class",
}

# Input layout
st.markdown("### 🔍 Pre-operative Patient Information")
user_input = {}
form_cols = st.columns(2)

for i, col in enumerate(input_vars):
    with form_cols[i % 2]:
        label = display_names.get(col, col)

        if col == "Albumin":
            val = st.number_input(label, value=4.0, step=0.1)
            user_input[col] = 1 if 3.5 <= val <= 5.49 else (2 if val < 3.5 else 3)

        elif col == "HCT":
            val = st.number_input(label, value=40.0, step=0.1)
            user_input[col] = 1 if 39 <= val <= 49 else (2 if val < 39 else 3)

        elif col == "BUN":
            val = st.number_input(label, value=15.0, step=0.1)
            user_input[col] = 1 if 5 <= val <= 20 else (2 if val < 5 else 3)

        elif col == "ASA":
            val = st.selectbox(label, ["I", "II", "III", "IV", "V"])
            asa_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
            user_input[col] = 1 if asa_map[val] in [1, 2] else 2

        elif col == "FHS":
            val = st.selectbox(label, ["Independent", "Dependent"])
            user_input[col] = 0 if val == "Independent" else 1

        else:
            # Binary categorical vars
            if set(label_encoders[col].classes_) == {"0", "1"} or set(label_encoders[col].classes_) == {0, 1}:
                val = st.selectbox(label, ["Yes", "No"])
                mapped_val = 1 if val == "Yes" else 0
            else:
                options = list(label_encoders[col].classes_)
                val = st.selectbox(label, options)
                mapped_val = label_encoders[col].transform([val])[0]

            user_input[col] = mapped_val

# Predict
st.markdown("---")
if st.button("🧠 Predict"):
    X_input = pd.DataFrame([user_input])
    y_pred = model.predict(X_input)[0]
    y_prob = model.predict_proba(X_input)[0][1]

    pred_label = "Yes" if y_pred == 1 else "No"
    color = "#ff4d4d" if y_pred == 1 else "#2ecc71"

    st.markdown(f"""
    <div style="background-color:white;border:1px solid #dee2e6;padding:1rem;border-radius:10px;">
        <h4>Complication Predicted?: <span style="color:{color};">{pred_label}</span></h4>
        
    </div>
    """, unsafe_allow_html=True)
# 📢 Disclaimer Section
st.markdown("---")
st.markdown("""
<div style="background-color:#FFF3CD; padding:15px; border-left:5px solid #FFA500;">
    <strong>Disclaimer Notice:</strong> 
    This risk calculator is provided for informational purposes only and is not intended to replace professional medical advice, diagnosis, or treatment. 
    The results are estimates based on general data and user-provided information, and may not accurately reflect individual outcomes. 
    Always consult your healthcare provider for personalized medical guidance.
</div>
""", unsafe_allow_html=True)
