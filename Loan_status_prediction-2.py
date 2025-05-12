import streamlit as st
import joblib
import os

# Load model and encoders with error handling
model_file = "loan_model.joblib"
encoder_file = "label_encoders.joblib"

if not os.path.exists(model_file) or not os.path.exists(encoder_file):
    st.error("❌ Required files not found: Ensure 'loan_model.joblib' and 'label_encoders.joblib' are uploaded with your app.")
    st.stop()
else:
    model = joblib.load(model_file)
    label_encoders = joblib.load(encoder_file)