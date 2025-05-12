import streamlit as st
import pandas as pd
import joblib

model = joblib.load("loan_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.title("Loan Approval Predictor")

# Dynamic input fields
user_input = {}
for col, le in label_encoders.items():
    options = le.classes_
    user_input[col] = st.selectbox(col, options)

# Add numeric fields based on the dataset
numerics = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
for num in numerics:
    user_input[num] = st.number_input(num, min_value=0.0)

# Predict
if st.button("Predict Loan Approval"):
    input_df = pd.DataFrame([user_input])
    for col, le in label_encoders.items():
        input_df[col] = le.transform([input_df[col][0]])
    prediction = model.predict(input_df)[0]
    st.success(f"Loan Status: {'Approved' if prediction == 1 else 'Rejected'}")