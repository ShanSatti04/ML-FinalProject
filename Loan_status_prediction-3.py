import streamlit as st
import pandas as pd
import joblib
import os

# Load model and encoders with proper path handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load(os.path.join("loan_model.joblib"))
        encoders = joblib.load(os.path.join("label_encoders.joblib"))
        return model, encoders
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        return None, None

model, label_encoders = load_model()

st.title("Loan Approval Predictor")

if model is not None and label_encoders is not None:
    user_input = {}
    for col, le in label_encoders.items():
        options = le.classes_
        user_input[col] = st.selectbox(col, options)

    numerics = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    for num in numerics:
        user_input[num] = st.number_input(num, min_value=0.0)

    if st.button("Predict Loan Approval"):
        input_df = pd.DataFrame([user_input])
        for col, le in label_encoders.items():
            input_df[col] = le.transform([input_df[col][0]])
        prediction = model.predict(input_df)[0]
        st.success(f"Loan Status: {'Approved' if prediction == 1 else 'Rejected'}")
else:
    st.warning("Model or encoders could not be loaded. Please check your deployment files.")
