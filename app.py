import streamlit as st
import pandas as pd
import joblib

# Load the model and encoder
model = joblib.load('logistic_model.pkl')
le = joblib.load('label_encoder.pkl')

st.title("Patient Death Prediction App")
st.write("Enter patient health details below:")

# Input fields for user
def user_input_features():
    usmr = st.number_input("USMR", min_value=1, max_value=20, value=1)
    medical_unit = st.number_input("Medical Unit", min_value=1, max_value=20, value=1)
    sex = st.selectbox("Sex", [1, 2])
    patient_type = st.selectbox("Patient Type", [1, 2])
    pneumonia = st.selectbox("Pneumonia", [1, 2])
    age = st.slider("Age", 0, 120, 30)
    pregnancy = st.selectbox("Pregnancy", [1, 2])
    diabetes = st.selectbox("Diabetes", [1, 2])
    copd = st.selectbox("COPD", [1, 2])
    asthma = st.selectbox("Asthma", [1, 2])
    inmsupr = st.selectbox("Immunosuppressed", [1, 2])
    hypertension = st.selectbox("Hypertension", [1, 2])
    cardiovascular = st.selectbox("Cardiovascular Disease", [1, 2])
    obesity = st.selectbox("Obesity", [1, 2])
    renal_chronic = st.selectbox("Chronic Renal Disease", [1, 2])
    tobacco = st.selectbox("Tobacco Use", [1, 2])

    data = {
        'usmr': usmr,
        'medical_unit': medical_unit,
        'sex': sex,
        'patient_type': patient_type,
        'pneumonia': pneumonia,
        'age': age,
        'pregnancy': pregnancy,
        'diabetes': diabetes,
        'copd': copd,
        'asthma': asthma,
        'inmsupr': inmsupr,
        'hypertension': hypertension,
        'cardiovascular': cardiovascular,
        'obesity': obesity,
        'renal_chronic': renal_chronic,
        'tobacco': tobacco
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    label = le.inverse_transform([prediction])[0]
    
    st.subheader("Prediction Result")
    st.write(f"Predicted Class: {int(prediction)} ({label})")
    st.write(f"Probability of Death: {proba:.2%}")
