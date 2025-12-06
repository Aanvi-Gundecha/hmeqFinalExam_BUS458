
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("/content/drive/MyDrive/FinalProj458/my_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #e6f7ff; padding: 10px; color: #004080;'><b>Loan Approval Prediction</b></h1>",
    unsafe_allow_html=True
)

# Numeric inputs
st.header("Enter Applicant's Financial Details and Demographics")

# Input fields for numeric values
granted_loan = st.slider("Granted Loan Amount", min_value=1000, max_value=500000, step=1000, value=50000)
requested_loan = st.slider("Requested Loan Amount", min_value=1000, max_value=500000, step=1000, value=50000)
fico_score = st.slider("FICO Score", min_value=300, max_value=850, step=1, value=650)
monthly_gross_income = st.slider("Monthly Gross Income", min_value=0, max_value=20000, step=100, value=5000)
monthly_housing_payment = st.slider("Monthly Housing Payment", min_value=0, max_value=10000, step=50, value=1500)

# Binary/Categorical inputs
ever_bankrupt = st.checkbox("Ever Bankrupt or Foreclosed?")

reason_options = ['credit_card_refinancing', 'debt_conslidation', 'home_improvement', 'major_purchase', 'other']
reason = st.selectbox("Reason for Loan", reason_options)

fico_group_options = ['poor', 'fair', 'good', 'very_good']
fico_group = st.selectbox("FICO Score Group", fico_group_options, index=1) # Default to 'fair'

employment_status_options = ['full_time', 'part_time', 'unemployed']
employment_status = st.selectbox("Employment Status", employment_status_options, index=0) # Default to 'full_time'

employment_sector_options = [
    'communication_services', 'consumer_discretionary', 'consumer_staples',
    'energy', 'financials', 'health_care', 'industrials',
    'information_technology', 'materials', 'real_estate', 'utilities', 'other_sector'
]
employment_sector = st.selectbox("Employment Sector", employment_sector_options, index=7) # Default to 'information_technology'

lender_options = ['A', 'B', 'C']
lender = st.selectbox("Preferred Lender", lender_options, index=0) # Default to 'A'


# Create the input data as a DataFrame
# Note: These columns are the "original" features before one-hot encoding
input_data = pd.DataFrame({
    "Granted_Loan_Amount": [granted_loan],
    "Requested_Loan_Amount": [requested_loan],
    "FICO_score": [fico_score],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Ever_Bankrupt_or_Foreclose": [1 if ever_bankrupt else 0],
    "Reason": [reason],
    "Fico_Score_group": [fico_group],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Lender": [lender]
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
input_data_encoded = pd.get_dummies(input_data, columns=[
    'Reason', 'Fico_Score_group', 'Employment_Status', 'Employment_Sector', 'Lender'
])

# 2. Add any "missing" columns the model expects (fill with 0).
# This step is crucial if the user selects a category that wasn't in the training data,
# or if get_dummies doesn't produce all expected OHE columns for a given run.
for col_name in model.feature_names_in_:
    if col_name not in input_data_encoded.columns:
        input_data_encoded[col_name] = 0

# 3. Reorder/filter columns to exactly match the model's training data.
input_data_encoded = input_data_encoded[model.feature_names_in_]

# Predict button
if st.button("Predict Loan Outcome"):
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]

    # Display result
    if prediction == 1:
        st.error("Prediction: **Likely to Default (Bad Loan)** 🚫")
        st.write("This applicant has a higher probability of defaulting on the loan.")
    else:
        st.success("Prediction: **Low Risk (Good Loan)** 💲")
        st.write("This applicant has a lower probability of defaulting on the loan.")
