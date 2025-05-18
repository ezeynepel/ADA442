import streamlit as st
import pandas as pd
import pickle

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="Term Deposit Tahmini",
    page_icon="💰",
    layout="centered"
)

# --- Model Yükleme ---
@st.cache_resource
def load_model():
    with open('best_model_1_decision_tree.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['selected_features']

model, selected_features = load_model()

# --- Başlık ve Giriş ---
st.title("📊 Banka Vadeli Mevduat Tahmini")
st.markdown("Bu uygulama, bir bankanın müşterisinin **vadeli mevduat (term deposit)** ürününü alıp almayacağını tahmin eder.")
st.info("📞 Veri seti, Portekiz'de yapılan bir tele-pazarlama kampanyasından alınmıştır.")

# --- Yan Menü Bilgi ---
with st.sidebar:
    st.header("📌 Proje Bilgisi")
    st.markdown("""
    - 👨‍💻 Model: Decision Tree  
    - 📈 Değerlendirme: F1, Accuracy  
    - 🔍 Özellik Sayısı: {}  
    - 👩‍🔬 Tahmin: 'yes' veya 'no'
    """.format(len(selected_features)))

# --- Kullanıcı Girdileri ---
st.subheader("🔍 Müşteri Bilgilerini Girin")

user_input = {}
for feature in selected_features:
    user_input[feature] = st.text_input(f"{feature}", placeholder="Değer girin...")

# --- Tahmin Butonu ---
if st.button("🔮 Tahmin Et"):
    try:
        input_df = pd.DataFrame([user_input])
        input_df = input_df.apply(pd.to_numeric, errors='ignore')
        prediction = model.predict(input_df)[0]

        if prediction == 'yes':
            st.success("✅ Bu müşteri yüksek ihtimalle **vadeli mevduat alacak.**")
        else:
            st.error("❌ Bu müşteri büyük ihtimalle **almayacak.**")
    except Exception as e:
        st.warning(f"Hata oluştu: {e}")
