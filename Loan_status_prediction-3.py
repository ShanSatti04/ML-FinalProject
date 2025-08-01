﻿import streamlit as st
import pandas as pd
import joblib
import os

# Load model and encoders
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(os.path.join("loan_model.joblib"))
        encoders = joblib.load(os.path.join("label_encoders.joblib"))
        return model, encoders
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        return None, None

# Load assets
model, label_encoders = load_assets()

# Define the exact feature order expected by the model
feature_names = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]


st.title("🏦 Loan Approval Predictor")

if model is not None and label_encoders is not None:
    st.subheader("Enter Applicant Information")

    user_input = {}

    # Categorical inputs
    for col, le in label_encoders.items():
        options = le.classes_
        user_input[col] = st.selectbox(f"{col}", options)

    # Numeric inputs
    numerics = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    for num in numerics:
        user_input[num] = st.number_input(f"{num}", min_value=0.0)

    if st.button("🔍 Predict Loan Approval"):
        input_data = {}

        # Encode categorical features
        for col, le in label_encoders.items():
            input_data[col] = le.transform([user_input[col]])[0]

        # Add numeric values directly
        for num in numerics:
            input_data[num] = user_input[num]

        # Create DataFrame in expected order
        input_df = pd.DataFrame([[input_data[col] for col in feature_names]], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df.values)[0]
        result = "✅ Approved" if prediction == 1 else "❌ Rejected"
        st.success(f"Loan Status: {result}")
else:
    st.warning("Model or encoders could not be loaded. Please check the files.")