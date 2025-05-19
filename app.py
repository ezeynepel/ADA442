import streamlit as st
import pickle
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# --- Model Yükleme Fonksiyonu ---
def load_model(model_path):
    with open(model_path, "rb") as file:
        loaded_object = pickle.load(file)
    if isinstance(loaded_object, dict):
        return loaded_object.get("model", None)
    return loaded_object

# --- Modeli Yükle ---
model_path = "final_model_with_pipeline.pkl"
model = load_model(model_path)

# --- Başlık ve Açıklama ---
st.set_page_config(page_title="Bank Term Deposit Prediction", layout="wide")
st.title("📈 ADA 442 | Term Deposit Prediction App")

st.markdown("""
This app predicts whether a client will subscribe to a term deposit based on personal and economic factors.

**Dataset**: [UCI Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
""")

# --- Girdi Başlığı ---
st.header("Enter Client Information")

# --- Input Formu ---
col1, col2, col3 = st.columns(3)

# Row 1
with col1:
    age = st.number_input("Age", min_value=18, max_value=120, step=1)
with col2:
    job = st.selectbox("Job", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician',
        'unemployed', 'unknown'
    ])
with col3:
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])

# Row 2
with col1:
    education = st.selectbox("Education", [
        'illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
        'high.school', 'professional.course', 'university.degree', 'unknown'
    ])
with col2:
    default = st.selectbox("Default Credit", ['yes', 'no', 'unknown'])
with col3:
    housing = st.selectbox("Housing Loan", ['yes', 'no', 'unknown'])

# Row 3
with col1:
    loan = st.selectbox("Personal Loan", ['yes', 'no', 'unknown'])
with col2:
    contact = st.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
with col3:
    month = st.selectbox("Last Contact Month", [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ])

# Row 4
with col1:
    day_of_week = st.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
with col2:
    duration = st.number_input("Call Duration (sec)", min_value=0, step=1)
with col3:
    campaign = st.number_input("Campaign Contact Count", min_value=0, step=1)

# Row 5
with col1:
    pdays = st.number_input("Days Since Last Contact", min_value=0, step=1)
with col2:
    previous = st.number_input("Number of Previous Contacts", min_value=0, step=1)
with col3:
    poutcome = st.selectbox("Previous Campaign Outcome", ['success', 'failure', 'nonexistent', 'unknown'])

# Row 6
with col1:
    emp_var_rate = st.number_input("Employment Variation Rate", step=0.01)
with col2:
    cons_price_idx = st.number_input("Consumer Price Index", step=0.01)
with col3:
    cons_conf_idx = st.number_input("Consumer Confidence Index", step=0.01)

# Row 7
with col1:
    euribor3m = st.number_input("Euribor 3 Month Rate", step=0.01)
with col2:
    nr_employed = st.number_input("Number of Employees", step=1)

# --- Predict Butonu ---
predict_button = st.button("🔮 Predict")

# --- Tahmin Bölümü ---
if predict_button:
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

    # 🩹 "unknown" değerleri NaN yap (imputer çalışsın)
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
