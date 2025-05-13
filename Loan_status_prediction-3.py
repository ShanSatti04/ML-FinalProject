import streamlit as st
import pandas as pd
import joblib
import os

# Load model, encoders, and feature names
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(os.path.join("loan_model.joblib"))
        encoders = joblib.load(os.path.join("label_encoders.joblib"))
        feature_names = joblib.load(os.path.join("model_features.joblib"))
        return model, encoders, feature_names
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        return None, None, None

model, label_encoders, feature_names = load_assets()

st.title("🏦 Loan Approval Predictor")

if model is not None and label_encoders is not None and feature_names is not None:
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

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([[input_data[col] for col in feature_names]], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "✅ Approved" if prediction == 1 else "❌ Rejected"
        st.success(f"Loan Status: {result}")
else:
    st.warning("Model, encoders, or feature names could not be loaded. Please ensure all required files are present.")
