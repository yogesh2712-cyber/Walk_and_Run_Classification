import streamlit as st
import numpy as np
import joblib
import os

# -------------------------------

# LOAD MODEL

# -------------------------------

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    return joblib.load(model_path)

model = load_model()

# -------------------------------

# PAGE CONFIG

# -------------------------------

st.set_page_config(page_title="Walk Run Classifier", layout="centered")

# -------------------------------

# TITLE

# -------------------------------

st.title("🏃 Walk vs Run Classification App")
st.write("Predict whether a person is Walking 🚶 or Running 🏃")

# -------------------------------

# SIDEBAR INPUT

# -------------------------------

st.sidebar.header("Enter Sensor Values")

acc_x = st.sidebar.number_input("Acceleration X", value=0.0)
acc_y = st.sidebar.number_input("Acceleration Y", value=0.0)
acc_z = st.sidebar.number_input("Acceleration Z", value=0.0)

gyro_x = st.sidebar.number_input("Gyroscope X", value=0.0)
gyro_y = st.sidebar.number_input("Gyroscope Y", value=0.0)
gyro_z = st.sidebar.number_input("Gyroscope Z", value=0.0)

# -------------------------------

# PREDICT

# -------------------------------

if st.button("Predict"):
    try:
        # Input must match training features
        input_data = np.array([[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]])

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # -------------------------------
        # RESULT
        # -------------------------------
        st.subheader("Result")

        if prediction == 1:
            st.success("🏃 Running")
        else:
            st.success("🚶 Walking")

        # -------------------------------
        # PROBABILITY
        # -------------------------------
        st.subheader("Confidence")

        st.progress(float(max(probability)))

        st.write(f"Walking: {probability[0]*100:.2f}%")
        st.write(f"Running: {probability[1]*100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")

