import streamlit as st
import numpy as np
import pickle

# Load the model and mappings
with open("model/RF_classifier.pkl", "rb") as model_file:
    RF_classifier = pickle.load(model_file)

with open("model/Crop_Mappings.pkl", "rb") as mapping_file:
    Crop_Mappings = pickle.load(mapping_file)

# Inverse mappings to decode label predictions
inverse_mappings = {v: k for k, v in Crop_Mappings.items()}

# Prediction Function
def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    input_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    prediction = RF_classifier.predict(input_values)
    return prediction[0]

# Streamlit UI
st.title("🌾 Crop Recommendation System")
st.write("Enter the soil and environmental parameters to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=1000, value=21)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=1000, value=26)
K = st.number_input("Potassium (K)", min_value=0, max_value=1000, value=27)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=27.003155)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=47.675254)
pH = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=5.699587)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=95.851183)

# Predict Button
if st.button("Predict Crop"):
    pred = predict_crop(N, P, K, temperature, humidity, pH, rainfall)
    if pred in inverse_mappings:
        st.success(f"✅ The best crop to cultivate is: **{inverse_mappings[pred]}**")
    else:
        st.error("❌ Sorry, we could not determine the best crop with the provided data.")
