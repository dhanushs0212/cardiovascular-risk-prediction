import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Cardiovascular Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Cardiovascular Risk Prediction System")
st.markdown("---")

# ------------------------------
# Load Model and Scaler
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cardio_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.write("Enter patient details below:")

# ------------------------------
# Input Fields
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 40)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    chest_pain_type = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    resting_bp_s = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar (0/1)", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    max_heart_rate = st.number_input("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])

st.markdown("---")

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("🔍 Predict Risk"):

    input_data = np.array([[age, sex, chest_pain_type, resting_bp_s,
                            cholesterol, fasting_blood_sugar, resting_ecg,
                            max_heart_rate, exercise_angina,
                            oldpeak, st_slope]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict probability
    prediction = model.predict(input_scaled)[0][0]
    risk_percentage = prediction * 100

    st.markdown("### 📊 Prediction Result")

    if prediction > 0.5:
        st.error(f"⚠ High Risk of Cardiovascular Disease")
        st.write(f"Risk Probability: **{risk_percentage:.2f}%**")
    else:
        st.success("✅ Low Risk of Cardiovascular Disease")
        st.write(f"Risk Probability: **{risk_percentage:.2f}%**")

st.markdown("---")
st.write("### 📈 Model Performance")
st.write("Accuracy: 89%")
st.write("Recall (Disease Detection): 92%")
st.write("Model Type: Deep Neural Network (DNN)")