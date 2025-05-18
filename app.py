import streamlit as st
import pandas as pd
import pickle

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Bank Term Deposit Tahmini", layout="centered", page_icon="💰")

# --- Modeli Yükle ---
@st.cache_resource
def load_model():
    with open('best_model_1_decision_tree.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data['model'], model_data['selected_features']

model, selected_features = load_model()

# --- Başlık ---
st.title("📊 Banka Vadeli Mevduat Tahmini")
st.write("Müşterinin vadeli mevduat alıp almayacağını tahmin eden makine öğrenmesi uygulaması.")

# --- Tahmin Formu ---
with st.form("prediction_form"):
    st.subheader("🧾 Müşteri Bilgileri Girişi")
    cols = st.columns(3)

    user_inputs = {}
    for i, feature in enumerate(selected_features):
        with cols[i % 3]:
            user_inputs[feature] = st.text_input(f"{feature}", placeholder="Değer girin...")

    submitted = st.form_submit_button("🔮 Tahmin Et")

# --- Tahmin İşlemi ---
if submitted:
    try:
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df.apply(pd.to_numeric, errors='ignore')

        prediction = model.predict(input_df)[0]
        prediction_proba = None
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(input_df)[0]

        if prediction == 'yes':
            st.success("✅ Bu müşteri **vadeli mevduat alacak gibi görünüyor.**")
        else:
            st.error("❌ Bu müşteri **almayacak gibi duruyor.**")

        if prediction_proba is not None:
            st.info(f"🔢 Tahmin olasılığı (yes): %{prediction_proba[1] * 100:.2f}")

        st.caption("📌 Not: Bu tahmin eğitim verisiyle eğitilmiş bir modelle yapılmıştır.")

    except Exception as e:
        st.warning(f"Hata oluştu: {e}")
