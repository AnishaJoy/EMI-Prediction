# =======================================
# 💸 EMI Eligibility & EMI Amount Predictor
# =======================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page Setup & Custom Styling
# -----------------------------
st.set_page_config(page_title="💖 EMI Predictor", page_icon="💸", layout="centered")

st.markdown("""
    <style>
    /* Background and font setup */
    body {
        background-color: #738bb0;
        font-family: 'Poppins', sans-serif;
        color: #e6edf3;
    }

    .stApp {
        background: linear-gradient(180deg, #738bb0 0%, #637a99 100%);
    }

    /* Title styles */
    h1 {
        color: #58a6ff;
        text-align: center;
        font-weight: 800;
        font-size: 2.4rem;
        padding-top: 10px;
        margin-bottom: 0.5rem;
    }

    /* Subtitles */
    h2, h3 {
        color: #9b9bff;
        text-align: center;
        font-weight: 600;
        margin-top: 15px;
    }

    /* Labels */
    label {
        color: #c9d1d9 !important;
        font-weight: 500 !important;
    }

    /* Input boxes */
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #1c2128 !important;
        color: #e6edf3 !important;
        border-radius: 10px !important;
        border: 1px solid #30363d !important;
        box-shadow: none !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6f42c1, #8a63d2);
        color: #ffffff;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: 600;
        padding: 0.6em 1em;
        margin-top: 15px;
        box-shadow: 0px 4px 10px rgba(108, 70, 255, 0.25);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #8a63d2, #9c7bf9);
        transform: translateY(-2px);
        box-shadow: 0px 6px 12px rgba(155, 122, 255, 0.4);
    }

    /* Divider line */
    hr {
        border: 1px solid #30363d;
        margin: 25px 0;
    }

    /* Result cards */
    .success-box {
        background-color: #161b22;
        color: #58a6ff;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin-top: 15px;
        box-shadow: 0px 3px 10px rgba(88, 166, 255, 0.25);
        border: 1px solid #30363d;
    }

    /* Fade animation */
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Centered smaller caption */
    .caption {
        text-align: center;
        color: #8b949e;
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)



# -----------------------------
# Load Models
# -----------------------------
classifier = joblib.load("best_classifier_model.pkl")
regressor = joblib.load("best_regression_model.pkl")

st.title("💖 EMI Eligibility & EMI Amount Predictor")
st.caption("A simple and friendly app to check your EMI eligibility 💸 and predict your maximum affordable EMI amount 💕")

# -----------------------------
# Input Fields
# -----------------------------
st.markdown("### 🧾 Personal & Employment Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", 18, 70, 30)
    gender = st.selectbox("🚻 Gender", ["Male", "Female"])
    marital_status = st.selectbox("💍 Marital Status", ["Single", "Married"])
    education = st.selectbox("🎓 Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    employment_type = st.selectbox("💼 Employment Type", ["Private", "Self-Employed", "Government"])
    company_type = st.selectbox("🏢 Company Type", ["Mnc", "Mid-Size", "Small", "Startup"])

with col2:
    house_type = st.selectbox("🏠 House Type", ["Own", "Rented", "Family"])
    years_of_employment = st.number_input("🗓️ Years of Employment", 0.0, 40.0, 3.0)
    family_size = st.number_input("👨‍👩‍👧‍👦 Family Size", 1, 10, 4)
    dependents = st.number_input("🧒 Dependents", 0, 8, 1)
    existing_loans = st.selectbox("💳 Existing Loans", [0, 1])
    credit_score = st.number_input("⭐ Credit Score", 300, 900, 700)

st.markdown("---")
st.markdown("### 💵 Financial Details")

col3, col4 = st.columns(2)

with col3:
    monthly_salary = st.number_input("💰 Monthly Salary (₹)", 5000, 200000, 50000)
    monthly_rent = st.number_input("🏡 Monthly Rent (₹)", 0, 100000, 10000)
    bank_balance = st.number_input("🏦 Bank Balance (₹)", 0, 200000, 25000)
    emergency_fund = st.number_input("🚨 Emergency Fund (₹)", 0, 200000, 15000)

with col4:
    current_emi_amount = st.number_input("💸 Current EMI (₹)", 0, 100000, 10000)
    requested_amount = st.number_input("📑 Requested Loan Amount (₹)", 5000, 1000000, 200000)
    requested_tenure = st.number_input("⏳ Requested Tenure (months)", 6, 120, 24)
    total_expenses = st.number_input("🧾 Total Monthly Expenses (₹)", 1000, 100000, 12000)

emi_scenario = st.selectbox("🎯 EMI Scenario", [
    "Personal Loan Emi", "E-Commerce Shopping Emi", "Education Emi",
    "Vehicle Emi", "Home Appliances Emi"
])

# -----------------------------
# Derived Feature Calculations
# -----------------------------
debt_to_income = current_emi_amount / (monthly_salary + 1)
expense_to_income = total_expenses / (monthly_salary + 1)
affordability_ratio = bank_balance / (requested_amount + 1)
emp_stability = years_of_employment / (age + 1)

# -----------------------------
# Data Preparation
# -----------------------------
input_dict = {
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "marital_status": 1 if marital_status == "Married" else 0,
    "education": ["High School", "Graduate", "Post Graduate", "Professional"].index(education),
    "monthly_salary": monthly_salary,
    "years_of_employment": years_of_employment,
    "monthly_rent": monthly_rent,
    "family_size": family_size,
    "dependents": dependents,
    "existing_loans": existing_loans,
    "current_emi_amount": current_emi_amount,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
    "debt_to_income": debt_to_income,
    "total_expenses": total_expenses,
    "expense_to_income": expense_to_income,
    "affordability_ratio": affordability_ratio,
    "emp_stability": emp_stability,
    "employment_type_Private": 1 if employment_type == "Private" else 0,
    "employment_type_Self-Employed": 1 if employment_type == "Self-Employed" else 0,
    "company_type_Mid-Size": 1 if company_type == "Mid-Size" else 0,
    "company_type_Mnc": 1 if company_type == "Mnc" else 0,
    "company_type_Small": 1 if company_type == "Small" else 0,
    "company_type_Startup": 1 if company_type == "Startup" else 0,
    "house_type_Own": 1 if house_type == "Own" else 0,
    "house_type_Rented": 1 if house_type == "Rented" else 0,
    "emi_scenario_Education Emi": 1 if emi_scenario == "Education Emi" else 0,
    "emi_scenario_Home Appliances Emi": 1 if emi_scenario == "Home Appliances Emi" else 0,
    "emi_scenario_Personal Loan Emi": 1 if emi_scenario == "Personal Loan Emi" else 0,
    "emi_scenario_Vehicle Emi": 1 if emi_scenario == "Vehicle Emi" else 0
}

input_df = pd.DataFrame([input_dict])

expected_features = [
    'age', 'gender', 'marital_status', 'education', 'monthly_salary',
    'years_of_employment', 'monthly_rent', 'family_size', 'dependents',
    'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure',
    'debt_to_income', 'total_expenses', 'expense_to_income',
    'affordability_ratio', 'emp_stability', 'employment_type_Private',
    'employment_type_Self-Employed', 'company_type_Mid-Size',
    'company_type_Mnc', 'company_type_Small', 'company_type_Startup',
    'house_type_Own', 'house_type_Rented', 'emi_scenario_Education Emi',
    'emi_scenario_Home Appliances Emi', 'emi_scenario_Personal Loan Emi',
    'emi_scenario_Vehicle Emi'
]

for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[expected_features]

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("💖 Check EMI Eligibility"):
    eligibility_pred = classifier.predict(input_df)[0]

    if eligibility_pred == 2:
        st.markdown('<div class="success-box">❌ You are <b>Not Eligible</b> for EMI based on your financial profile.</div>', unsafe_allow_html=True)
    elif eligibility_pred == 1:
        st.markdown('<div class="success-box">⚠️ You are in the <b>High Risk</b> category. EMI approval may be uncertain.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ Congratulations! You are <b>Eligible</b> for EMI.</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📊 Predicted EMI Amount (₹)")
        emi_pred = regressor.predict(input_df)[0]
        st.success(f"💰 You can safely afford an EMI of **₹{emi_pred:,.2f}** per month.")
        st.balloons()
