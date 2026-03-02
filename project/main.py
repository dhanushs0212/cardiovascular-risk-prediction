import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load saved model and scaler
model = tf.keras.models.load_model("cardio_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Cardiovascular Risk Prediction")

st.write("Enter patient details below:")

# Input fields
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
chest_pain_type = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
resting_bp_s = st.number_input("Resting Blood Pressure")
cholesterol = st.number_input("Cholesterol")
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar (0/1)", [0,1])
resting_ecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
max_heart_rate = st.number_input("Max Heart Rate")
exercise_angina = st.selectbox("Exercise Induced Angina (0/1)", [0,1])
oldpeak = st.number_input("Oldpeak (ST depression)")
st_slope = st.selectbox("ST Slope (0-2)", [0,1,2])

if st.button("Predict Risk"):

    input_data = np.array([[age, sex, chest_pain_type, resting_bp_s,
                            cholesterol, fasting_blood_sugar, resting_ecg,
                            max_heart_rate, exercise_angina,
                            oldpeak, st_slope]])

    # Scale input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)[0][0]

    if prediction > 0.5:
        st.error(f"⚠ High Risk of Cardiovascular Disease ({prediction*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({(1-prediction)*100:.2f}%)")
