# ADA442
# Bank Term Deposit Predictor

This is a **Streamlit-based web application** developed for the course **ADA 442 - Statistical Learning** at **TED University**. 
The goal of the project is to predict whether a bank client will subscribe to a term deposit based on features obtained through direct marketing campaigns.

---

### ▶️ Try It Online  
🔗 [https://ada442-project.streamlit.app/](https://ada442-project.streamlit.app/)

---

## Model Overview

- **Model Type**: Logistic Regression
- **Pipeline Steps**:
  - Missing value imputation
  - One-Hot Encoding / Ordinal Encoding
  - Standard scaling (numerical features)
  - Feature selection (`VarianceThreshold`, `SelectKBest`)
- **Hyperparameter Tuning**: GridSearchCV with 5-fold Stratified Cross Validation


---

## Files & Structure

| File / Folder            | Description |
|--------------------------|-------------|
| `app.py`                 | Main Streamlit application |
| `final_model_with_pipeline.pkl` | Pre-trained model pipeline |
| `bank-additional.csv`    | Dataset from UCI ML repository |
| `README.md`              | Project documentation |

---

## Features

- Clean, multi-column input form
- Handles categorical values including 'unknown'
- Displays prediction result (yes/no)
- Visual effects (`st.balloons()`, `st.snow()`)
- Optional: prediction probability, feature importance, metrics

---

## Dataset Source
- [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## Course Info
> This application was developed as part of the **ADA 442 - Statistical Learning** course at **TED University**.

