import streamlit as st
import pandas as pd
import pickle

# --- Page Config ---
st.set_page_config(page_title="Bank Term Deposit Prediction", layout="centered", page_icon="💰")

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('best_model_1_decision_tree.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data['model'], model_data['selected_features']

model, selected_features = load_model()

# --- Title ---
st.title("📊 Bank Term Deposit Prediction")
st.write("Predict whether a client will subscribe to a term deposit using machine learning.")

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("🧾 Client Information Input")
    cols = st.columns(3)

    user_inputs = {}
    for i, feature in enumerate(selected_features):
        with cols[i % 3]:
            user_inputs[feature] = st.text_input(f"{feature}", placeholder="Enter a value...")

    submitted = st.form_submit_button("🔮 Predict")

# --- Prediction ---
if submitted:
    try:
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df.apply(pd.to_numeric, errors='ignore')

        prediction = model.predict(input_df)[0]
        prediction_proba = None
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(input_df)[0]

        if prediction == 'yes':
            st.success("✅ The client is likely to **subscribe** to a term deposit.")
        else:
            st.error("❌ The client is **not likely** to subscribe.")

        if prediction_proba is not None:
            st.info(f"🔢 Probability of 'yes': **{prediction_proba[1] * 100:.2f}%**")

        st.caption("📌 Note: This prediction is based on a pre-trained model using past campaign data.")

    except Exception as e:
        st.warning(f"⚠️ An error occurred: {e}")
