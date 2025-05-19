import streamlit as st
import pickle
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# Function to load the trained model
def load_model(model_path):
    with open(model_path, "rb") as file:
        loaded_object = pickle.load(file)
    # If the loaded object is a dictionary, extract the model
    if isinstance(loaded_object, dict):
        return loaded_object.get("model", None)
    return loaded_object


# Load the selected model
model_folder = "models"
model_path = "final_model_with_pipeline.pkl"
model = load_model(model_path)

st.title("ADA 442 Statistical Learning | Classification")
st.markdown("""

### Final Project Assignment: Bank Marketing Data Classification

**Objective**  
The objective of this project is to build a machine learning model to predict whether a client of a bank will subscribe to a term deposit or not. The dataset used for this project is the Bank Marketing Data Set, which can be found at [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (`variable y`). The marketing campaigns were based on phone calls. Often, more than one contact with the same client was required to assess if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

""")

# Header for input section
st.header("Enter Prediction Parameters")

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Row 1: Personal Information (Age, Job, Marital Status)
with col1:
    age = st.number_input("Age", min_value=18, max_value=120, step=1, help="Enter your age.")
with col2:
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", 
                               "management", "retired", "self-employed", "services", 
                               "student", "technician", "unemployed", "unknown"], 
                      help="Select your occupation.")
with col3:
    marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"], 
                           help="Select your marital status.")

# Row 2: Education, Default Credit, Housing Loan
with col1:
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"], 
                             help="Select your education level.")
with col2:
    default = st.selectbox("Default Credit", ["yes", "no", "unknown"], help="Do you have a default credit?")
with col3:
    housing = st.selectbox("Housing Loan", ["yes", "no", "unknown"], help="Do you have a housing loan?")

# Row 3: Personal Loan, Contact, Month
with col1:
    loan = st.selectbox("Personal Loan", ["yes", "no", "unknown"], help="Do you have a personal loan?")
with col2:
    contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"], 
                           help="Preferred contact communication type.")
with col3:
    month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", 
                                   "aug", "sep", "oct", "nov", "dec"], 
                         help="Select the month of the year.")

# Row 4: Day of Week, Duration, Campaign
with col1:
    day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"], help="Day of the week.")
with col2:
    duration = st.number_input("Duration (in seconds)", min_value=0, step=1, help="Duration of the last contact.")
with col3:
    campaign = st.number_input("Campaign", min_value=0, step=1, help="Number of contacts during the current campaign.")

# Row 5: Pdays, Previous Contacts, Outcome
with col1:
    pdays = st.number_input("Pdays (days since last contact)", min_value=0, step=1, help="Days since last contact.")
with col2:
    previous = st.number_input("Previous (contacts before)", min_value=0, step=1, 
                               help="Number of previous contacts.")
with col3:
    poutcome = st.selectbox("Outcome of previous campaign", ["success", "failure", "nonexistent", "unknown"], 
                            help="Outcome of the last marketing campaign.")

# Row 6: Empirical Variation Rate, Consumer Price Index, Consumer Confidence Index
with col1:
    emp_var_rate = st.number_input("Empirical Variation Rate", step=0.01, help="Empirical variation rate.")
with col2:
    cons_price_idx = st.number_input("Consumer Price Index", step=0.01, help="Consumer price index.")
with col3:
    cons_conf_idx = st.number_input("Consumer Confidence Index", step=0.01, help="Consumer confidence index.")

# Row 7: Euribor 3 Month Rate, Number of Employees
with col1:
    euribor3m = st.number_input("Euribor 3 Month Rate", step=0.01, help="Euribor 3-month rate.")
with col2:
    nr_employed = st.number_input("Number of Employees", step=1, help="Number of employees in the company.")
with col3:
    pass  # Placeholder for symmetry, no additional input here.

# --- Tahmin ---
if submit:
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }

    input_df = pd.DataFrame([input_dict])

    # 🩹 Categorical veride "unknown" varsa NaN olarak işaretle (imputer düzgün çalışsın)
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = input_df[col].replace("unknown", pd.NA)

    with st.spinner("Making prediction..."):
        try:
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            if prediction == 'yes':
                st.success(f"✅ The client is LIKELY to subscribe. (Probability: {prob:.2%})")
            else:
                st.error(f"❌ The client is UNLIKELY to subscribe. (Probability: {prob:.2%})")
        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")
