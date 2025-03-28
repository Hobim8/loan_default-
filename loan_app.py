import pandas as pd 
import sklearn
import streamlit as st
import numpy as np
import joblib

model = joblib.load('loan_default__model.pkl')

st.title("Loan Eligibility Checker")
st.subheader("Predict the likelihood of loan approval based on applicant information.")

# Create form
with st.form("loan_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.number_input("Dependents", min_value=0, max_value=100)
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0, max_value=1000000, step=100)
    LoanAmount = st.number_input("Loan Amount", min_value=0, max_value=1000000, step=1000)
    LoanAmountTerm = st.number_input("Loan Amount Term", min_value=0, max_value=10000000, step=10)
    CreditHistory = st.selectbox("Credit History", ["Good Credit History", "Bad"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    # Submit button inside the form
    submitted = st.form_submit_button("Predict Loan Status")

# Handle submission
if submitted:
    # Create input data
    input_data = pd.DataFrame([{
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': LoanAmountTerm,
        'Credit_History': CreditHistory,
        'Property_Area': Property_Area
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ Loan is likely to be Approved\n\nConfidence: {probability:.2f}")
    else:
        st.error(f"❌ Loan is likely to be Not Approved\n\nConfidence: {1 - probability:.2f}")
