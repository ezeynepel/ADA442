import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open('best_model_1_decision_tree.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['selected_features']

model, selected_features = load_model()

st.title("💰 Banka Vadeli Mevduat Tahmini")
st.write("Müşterinin vadeli mevduat alıp almayacağını tahmin eden bir makine öğrenmesi uygulaması.")

user_input = {}
for feature in selected_features:
    user_input[feature] = st.text_input(f"{feature} giriniz:")

if st.button("Tahmin Et"):
    try:
        input_df = pd.DataFrame([user_input])
        input_df = input_df.apply(pd.to_numeric, errors='ignore')
        prediction = model.predict(input_df)[0]
        st.success(f"Tahmin Sonucu: {prediction}")
    except Exception as e:
        st.error(f"Hata oluştu: {e}")
