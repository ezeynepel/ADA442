import streamlit as st
import pickle
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# --- Sayfa Ayarı ---
st.set_page_config(page_title="Bank Term Deposit Prediction", layout="wide")
st.title("📈 Bank Term Deposit Prediction App")
st.markdown("""
This app predicts whether a client will subscribe to a term deposit, based on their information and economic indicators.
""")

# --- Model Yükleme Fonksiyonu ---
def load_model(model_path):
    with open(model_path, "rb") as file:
        loaded_object = pickle.load(file)
    if isinstance(loaded_object, dict):
        return loaded_object.get("model", None)
    return loaded_object

# --- Model Seçimi ve Bilgi Gösterimi (Sidebar) ---
st.sidebar.title("Select Model")
models_available = {
    "Model 1: Decision Tree": "best_model_1_decision_tree.pkl",
    "Model 2: Logistic Regression (Robust Scaler)": "best_model_2_log_reg_RobustScaler.pkl",
    "Model 3: Logistic Regression (Standard Scaler)": "best_model_3_log_reg_StandardScaler.pkl",
}
selected_model_name = st.sidebar.selectbox("Choose a model", list(models_available.keys()))

model_info = {
    "Model 1: Decision Tree": """
    ### Model 1 - Decision Tree
    - **Max Depth**: 5
    - **Accuracy**: 91.08%
    - **F1 Score**: 0.9082
    """,
    "Model 2: Logistic Regression (Robust Scaler)": """
    ### Model 2 - Logistic Regression (Robust Scaler)
    - **C**: 10, Penalty: l2, Solver: lbfgs
    - **Accuracy**: 91.62%
    - **F1 Score**: 0.9007
    """,
    "Model 3: Logistic Regression (Standard Scaler)": """
    ### Model 3 - Logistic Regression (Standard Scaler)
    - **C**: 10, Penalty: l2, Solver: lbfgs
    - **Accuracy**: 91.75%
    - **F1 Score**: 0.8997
    """
}
st.sidebar.markdown(model_info[selected_model_name])

# Modeli yükle
model_path = os.path.join("models", models_available[selected_model_name])
model = load_model(model_path)

# --- Kullanıcıdan Girdi Alma (Form ile) ---
with st.form(key="user_input_form"):
    st.header("🧾 Client Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                                   'retired', 'self-employed', 'services', 'student', 'technician',
                                   'unemployed', 'unknown'])
        marital = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
        education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])

    with col2:
        default = st.selectbox("Has Default Credit?", ['yes', 'no'])
        housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
        loan = st.selectbox("Has Personal Loan?", ['yes', 'no'])
        contact = st.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])

    with col3:
        month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        day_of_week = st.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
        duration = st.number_input("Last Call Duration (sec)", min_value=0, value=100)

    st.subheader("📊 Economic & Campaign Indicators")
    col4, col5, col6 = st.columns(3)

    with col4:
        campaign = st.number_input("Campaign Contact Count", min_value=1, value=1)
        pdays = st.number_input("Days Since Last Contact", min_value=-1, value=999)

    with col5:
        previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)
        poutcome = st.selectbox("Previous Campaign Outcome", ['failure', 'nonexistent', 'success'])

    with col6:
        emp_var_rate = st.number_input("Employment Variation Rate", value=0.0)
        cons_price_idx = st.number_input("Consumer Price Index", value=93.0)
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)

    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.0)
    nr_employed = st.number_input("Number of Employees", value=5000.0)

    submit = st.form_submit_button("🔮 Predict")

# --- Tahmin Kısmı ---
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

    # Eksik kolonları tamamla
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    with st.spinner("Making prediction..."):
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

    if prediction == 'yes':
        st.success(f"✅ The client is LIKELY to subscribe. (Probability: {prob:.2%})")
    else:
        st.error(f"❌ The client is UNLIKELY to subscribe. (Probability: {prob:.2%})")
