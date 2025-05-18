import streamlit as st
import pandas as pd
import pickle

# Sayfa Ayarları
st.set_page_config(page_title="Bank Term Deposit Tahmini", page_icon="💰", layout="centered")

# Model Yükleme
@st.cache_resource
def load_model():
    with open('best_model_1_decision_tree.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['selected_features']

model, selected_features = load_model()

# Giriş
st.title("📊 Banka Vadeli Mevduat Tahmini")
st.markdown("Bir müşterinin **vadeli mevduat** ürünü alıp almayacağını makine öğrenmesi ile tahmin edin.")
st.caption("Model: Decision Tree | Proje: ADA442 - Statistical Learning")

# Yan Panel Bilgileri
with st.sidebar:
    st.header("📌 Bilgi")
    st.markdown("""
    - Model tipi: Decision Tree  
    - F1 Score: **0.91**  
    - Verisetinden 20 özellik seçildi  
    - Tahmin sonucu: 'yes' ya da 'no'
    """)
    st.markdown("---")
    st.markdown("👩‍💻 Hazırlayan: Elif")

# Kullanıcıdan Giriş
st.subheader("🧾 Müşteri Bilgilerini Girin")

user_input = {}
for feature in selected_features:
    user_input[feature] = st.text_input(f"{feature}", placeholder="Değer girin...")

# Tahmin Butonu
if st.button("🔮 Tahmin Et"):
    try:
        input_df = pd.DataFrame([user_input])
        input_df = input_df.apply(pd.to_numeric, errors='ignore')
        prediction = model.predict(input_df)[0]

        if prediction == 'yes':
            st.success("✅ Bu müşteri **vadeli mevduat alacak gibi görünüyor.**")
        else:
            st.error("❌ Bu müşteri **almayacak gibi duruyor.**")

        st.caption("📌 Not: Bu tahmin eğitim verisine göre yapılmıştır.")

    except Exception as e:
        st.warning(f"Hata oluştu: {e}")
