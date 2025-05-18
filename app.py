import streamlit as st
import pandas as pd
import pickle

# --- Page Settings ---
st.set_page_config(page_title="Term Deposit Prediction", layout="centered", page_icon="💰")

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('best_model_1_decision_tree.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data['model'], model_data['selected_features']

model, selected_features = load_model()

# --- App Title ---
st.title("🏦 Term Deposit Subscription Prediction")
st.markdown("Fill in the client's information below to predict whether they are likely to **subscribe** to a bank term deposit.")

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("📋 Client Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        job = st.selectbox("Occupation", [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"
        ])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
        education = st.selectbox("Education Level", [
            "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
            "professional.course", "university.degree", "unknown"
        ])
        default = st.selectbox("Has Credit in Default?", ["yes", "no", "unknown"])

    with col2:
        housing = st.selectbox("Has Housing Loan?", ["yes", "no", "unknown"])
        loan = st.selectbox("Has Personal Loan?", ["yes", "no", "unknown"])
        contact = st.selectbox("Contact Communication Type", ["cellular", "telephone"])
        month = st.selectbox("Last Contact Month", [
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        day_of_week = st.selectbox("Last Contact Day", ["mon", "tue", "wed", "thu", "fri"])

    with col3:
        duration = st.slider("Last Call Duration (sec)", min_value=0, max_value=5000, step=10, value=100)
        campaign = st.slider("Campaign Contact Count", min_value=1, max_value=50, step=1, value=1)
        pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=999, value=999)
        previous = st.number_input("Previous Campaign Contacts", min_value=0, max_value=50, value=0)
        poutcome = st.selectbox("Previous Campaign Outcome", ["success", "failure", "nonexistent", "unknown"])

    st.subheader("📊 Economic Indicators")
    col4, col5, col6 = st.columns(3)

    with col4:
        emp_var_rate = st.number_input("Employment Variation Rate (%)", value=0.0, step=0.01)
    with col5:
        cons_price_idx = st.number_input("Consumer Price Index", value=93.0, step=0.01)
    with col6:
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0, step=0.1)

    col7, col8 = st.columns(2)
    with col7:
        euribor3m = st.number_input("Euribor 3-Month Rate", value=4.0, step=0.01)
    with col8:
        nr_employed = st.number_input("Number of Employees in the Company", value=5000.0, step=1.0)

    submit = st.form_submit_button("🔮 Predict")

# --- Prediction Output ---
if submit:
    input_data = pd.DataFrame([{
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
    }])

    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = None
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(input_data)[0]

        if prediction == 'yes':
            st.success("✅ The client is likely to **subscribe** to a term deposit.")
        else:
            st.error("❌ The client is **not likely** to subscribe.")

        if prediction_proba is not None:
            st.info(f"🔢 Probability of 'yes': **{prediction_proba[1] * 100:.2f}%**")

        st.caption("📌 Note: This prediction is based on past marketing campaign data.")

    except Exception as e:
        st.warning(f"⚠️ Error occurred: {e}")
