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

# Sidebar for model selection
st.sidebar.title("Select Model")
models_available = {
    "Model 1: Decision Tree": "best_model_1_decision_tree.pkl",
    "Model 2: Logistic Regression (Robust Scaler)": "best_model_2_log_reg_RobustScaler.pkl",
    "Model 3: Logistic Regression (Standard Scaler)": "best_model_3_log_reg_StandardScaler.pkl",
}
selected_model_name = st.sidebar.selectbox("Choose a model", list(models_available.keys()))

# Display model information below the model selection part
model_info = {
    "Model 1: Decision Tree": """
    ### Model Information: Model 1
    - **Type**: Decision Tree Classifier
    - **Details**: 
      - Max Depth: 5
      - Min Samples Split: 2
    - **Accuracy**: 0.9108 (Cross-validated)
    - **F1 Score**: 0.9082
    - **Precision**: 0.9120
    - **Recall**: 0.9053
    """,
    "Model 2: Logistic Regression (Robust Scaler)": """
    ### Model Information: Model 2
    - **Type**: Logistic Regression
    - **Scaler**: Robust Scaler
    - **Details**: 
      - C: 10
      - Penalty: l2
      - Solver: lbfgs
    - **Accuracy**: 0.9162 (Cross-validated)
    - **F1 Score**: 0.9007
    - **Precision**: 0.8980
    - **Recall**: 0.9090
    """,
    "Model 3: Logistic Regression (Standard Scaler)": """
    ### Model Information: Model 3
    - **Type**: Logistic Regression
    - **Scaler**: Standard Scaler
    - **Details**: 
      - C: 10
      - Penalty: l2
      - Solver: lbfgs
    - **Accuracy**: 0.9175 (Cross-validated)
    - **F1 Score**: 0.8997
    - **Precision**: 0.8968
    - **Recall**: 0.9078
    """
}


st.sidebar.markdown(model_info[selected_model_name])

# Load the selected model
model_folder = "models"
model_path = os.path.join(model_folder, models_available[selected_model_name])
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

# Collecting all the inputs into a dictionary
input_features_raw = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
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
}

# Convert input to a DataFrame and prepare for prediction
input_df = pd.DataFrame([input_features_raw])

# Dynamically adjust columns to match the model's expected input
missing_cols = set(model.feature_names_in_) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

# Reorder columns to match the model's expected input
input_df = input_df[model.feature_names_in_]

# Prediction button
predict_button = st.button("Predict", key="predict", help="Click here to make a prediction")

if predict_button:
    with st.spinner("Making prediction..."):
        try:
            # Make the prediction
            prediction = model.predict(input_df)
            st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
