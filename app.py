import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the model
with open('RF_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define column names used during training
# feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
# Correct feature names used during model training
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']



# Crop dictionary for prediction
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Streamlit UI
st.title("🌱 Crop Recommendation System")
st.write("Enter the soil parameters below to predict the best crop for cultivation.")

# Create input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=300, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=300, value=30)
K = st.number_input("Potassium (K)", min_value=0, max_value=300, value=40)
temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

# Button to predict
if st.button("🌾 Predict Crop"):
    # Prepare input data
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    
    # Convert to DataFrame with correct feature names
    # single_pred = pd.DataFrame([feature_list], columns=feature_names)
    # Convert to DataFrame with correct feature names
    single_pred = pd.DataFrame([feature_list], columns=feature_names)


    # Make prediction
    prediction = model.predict(single_pred)[0]
    crop = crop_dict.get(prediction, "Unknown")

    # Display result
    st.success(f"✅ Recommended Crop: **{crop}** is the best crop to be cultivated right there.")
