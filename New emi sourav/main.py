import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------
# Load saved models
# -------------------------------------------------------
classifier = joblib.load("best_classifier_model.pkl")
regressor = joblib.load("best_regression_model.pkl")

st.set_page_config(page_title="💸 EMI Predictor", page_icon="💰", layout="wide")

st.title("💰 EMI Eligibility & EMI Amount Prediction")
st.write("""
This app predicts whether you are **eligible** for EMI and,  
if eligible, estimates your **maximum safe EMI amount** per month.
""")

# -------------------------------------------------------
# UI Layout
# -------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Private", "Self-Employed", "Government"])
    company_type = st.selectbox("Company Type", ["Midsize", "MNC", "Small", "Startup"])
    house_type = st.selectbox("House Type", ["Own", "Rented"])

with col2:
    monthly_salary = st.number_input("Monthly Salary (₹)", 5000, 200000, 50000)
    years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 3.0)
    monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 10000)
    family_size = st.number_input("Family Size", 1, 15, 4)
    dependents = st.number_input("Dependents", 0, 10, 1)
    existing_loans = st.selectbox("Existing Loans?", [0, 1])
    current_emi_amount = st.number_input("Current EMI (₹)", 0, 100000, 10000)
    credit_score = st.number_input("Credit Score", 300, 900, 700)

st.subheader("Additional Financial Inputs")
col3, col4 = st.columns(2)

with col3:
    school_fees = st.number_input("School Fees (₹)", 0, 100000, 5000)
    college_fees = st.number_input("College Fees (₹)", 0, 100000, 5000)
    travel_expenses = st.number_input("Travel Expenses (₹)", 0, 50000, 2000)
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 50000, 8000)

with col4:
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 50000, 3000)
    bank_balance = st.number_input("Bank Balance (₹)", 0, 500000, 20000)
    emergency_fund = st.number_input("Emergency Fund (₹)", 0, 500000, 15000)

emi_scenario = st.selectbox("EMI Scenario", [
    "Personal Loan Emi",
    "Ecommerce Shopping Emi",
    "Education Emi",
    "Vehicle Emi",
    "Home Appliances Emi"
])

requested_amount = st.number_input("Requested Loan Amount (₹)", 5000, 2000000, 200000)
requested_tenure = st.number_input("Requested Tenure (Months)", 6, 120, 24)

# -------------------------------------------------------
# Feature Engineering matching training
# -------------------------------------------------------
total_expenses = (
    school_fees +
    college_fees +
    travel_expenses +
    groceries_utilities +
    other_monthly_expenses
)

disposable_income = monthly_salary - total_expenses
emi_to_salary_ratio = current_emi_amount / monthly_salary if monthly_salary > 0 else 0
dti = current_emi_amount / monthly_salary if monthly_salary > 0 else 0
savings_ratio = bank_balance / monthly_salary if monthly_salary > 0 else 0
emergency_ratio = emergency_fund / monthly_salary if monthly_salary > 0 else 0
bank_balance_ratio = bank_balance / monthly_salary if monthly_salary > 0 else 0

# Salary bin (same as qcut in training)
# salary_bin_2 = 1 if monthly_salary > df["monthly_salary"].quantile(0.25) else 0
# salary_bin_3 = 1 if monthly_salary > df["monthly_salary"].quantile(0.50) else 0
# salary_bin_4 = 1 if monthly_salary > df["monthly_salary"].quantile(0.75) else 0

# -------------------------------------------------------
# Manual Encoding (MUST match training dummies)
# -------------------------------------------------------
input_dict = {
    # Base numeric features
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "marital_status": 0 if marital_status == "Married" else 1,
    "education": ["High School", "Graduate", "Post Graduate", "Professional"].index(education) + 1,
    "monthly_salary": monthly_salary,
    "years_of_employment": years_of_employment,
    "monthly_rent": monthly_rent,
    "family_size": family_size,
    "dependents": dependents,
    "school_fees": school_fees,
    "college_fees": college_fees,
    "travel_expenses": travel_expenses,
    "groceries_utilities": groceries_utilities,
    "other_monthly_expenses": other_monthly_expenses,
    "existing_loans": existing_loans,
    "current_emi_amount": current_emi_amount,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,

    # Engineered features
    "total_expenses": total_expenses,
    "disposable_income": disposable_income,
    "emi_to_salary_ratio": emi_to_salary_ratio,
    "dti": dti,
    "savings_ratio": savings_ratio,
    "emergency_ratio": emergency_ratio,
    "bank_balance_ratio": bank_balance_ratio,

    # Encoded categories (match df.columns)
    "employment_type_private": 1 if employment_type == "Private" else 0,
    "employment_type_self-employed": 1 if employment_type == "Self-Employed" else 0,

    "company_type_midsize": 1 if company_type == "Midsize" else 0,
    "company_type_mnc": 1 if company_type == "MNC" else 0,
    "company_type_small": 1 if company_type == "Small" else 0,
    "company_type_startup": 1 if company_type == "Startup" else 0,

    "house_type_own": 1 if house_type == "Own" else 0,
    "house_type_rented": 1 if house_type == "Rented" else 0,

    "emi_scenario_education emi": 1 if emi_scenario == "Education Emi" else 0,
    "emi_scenario_home appliances emi": 1 if emi_scenario == "Home Appliances Emi" else 0,
    "emi_scenario_personal loan emi": 1 if emi_scenario == "Personal Loan Emi" else 0,
    "emi_scenario_vehicle emi": 1 if emi_scenario == "Vehicle Emi" else 0,

    # "salary_bin_2": salary_bin_2,
    # "salary_bin_3": salary_bin_3,
    # "salary_bin_4": salary_bin_4,
}

input_df = pd.DataFrame([input_dict])

# Ensure all model-required columns exist
model_features = classifier.get_booster().feature_names
for col in model_features:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[model_features]

# -------------------------------------------------------
# Prediction Handling
# -------------------------------------------------------
st.markdown("---")
if st.button("🔍 Check EMI Eligibility"):
    
    eligibility = classifier.predict(input_df)[0]

    if eligibility == 2:
        st.error("❌ You are **Not Eligible** for EMI.")
    elif eligibility == 1:
        st.warning("⚠️ You are in a **High Risk** category.")
    else:
        st.success("✅ You are **Eligible for EMI**!")
        st.markdown("### 📊 Estimated EMI You Can Afford")

        emi = regressor.predict(input_df)[0]
        st.success(f"💸 Maximum Safe EMI: **₹{emi:,.2f}** per month")
        st.balloons()
