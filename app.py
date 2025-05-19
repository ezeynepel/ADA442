import streamlit as st
import pickle
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# --- Load final model trained in code.ipynb ---
@st.cache_resource
def load_model():
    with open("final_model_with_pipeline.pkl", "rb") as file:
        loaded_data = pickle.load(file)
        return loaded_data["model"] if isinstance(loaded_data, dict) else loaded_data

model = load_model()

# --- Title and description ---
st.title("ADA 442 | Final Project – Bank Term Deposit Prediction")

# --- Input fields ---
st.header("Client Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur", "housemaid", 
        "management", "retired", "self-employed", "services", 
        "student", "technician", "unemployed"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", [
        "illiterate", "basic.4y", "basic.6y", "basic.9y", 
        "high.school", "professional.course", "university.degree"])
    default = st.selectbox("Default Credit", ["yes", "no"])

with col2:
    housing = st.selectbox("Housing Loan", ["yes", "no"])
    loan = st.selectbox("Personal Loan", ["yes", "no"])
    emp_var_rate = st.number_input("Empirical Var. Rate", step=0.01)
    cons_price_idx = st.number_input("Consumer Price Index", step=0.01) 
    cons_conf_idx = st.number_input("Consumer Confidence Index", step=0.01)
    
with col3:    
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
    duration = st.number_input("Last Call Duration", step=1)
    campaign = st.number_input("Campaign Contacts", step=1)
    
with col4:
    pdays = st.number_input("Days Since Last Contact", step=1)
    previous = st.number_input("Previous Contacts", step=1)
    poutcome = st.selectbox("Previous Outcome", ["success", "failure", "nonexistent"])
    euribor3m = st.number_input("Euribor 3 Month", step=0.01)
    nr_employed = st.number_input("Number of Employees", step=1)
   
    

# --- Prepare input ---
ordinal_edu = {
    'illiterate': 0,
    'basic.4y': 1,
    'basic.6y': 2,
    'basic.9y': 3,
    'high.school': 4,
    'professional.course': 5,
    'university.degree': 6
}

input_data = pd.DataFrame([{ 
    "age": age,
    "job": job,
    "marital": marital,
    "education": ordinal_edu[education],
    "default": default,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "month": month,
    "day_of_week": day_of_week,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
    "emp.var.rate": emp_var_rate,
    "cons.price.idx": cons_price_idx,
    "cons.conf.idx": cons_conf_idx,
    "euribor3m": euribor3m,
    "nr.employed": nr_employed
}])

# --- Prediction ---
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Prediction: {'Yes' if prediction[0] == 'yes' else 'No'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
