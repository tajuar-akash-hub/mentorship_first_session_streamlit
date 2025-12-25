import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("🏠 House Price Prediction")

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Input fields
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", value=5000)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6])
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
    stories = st.selectbox("Stories", [1, 2, 3, 4])
    parking = st.selectbox("Parking", [0, 1, 2, 3])

with col2:
    mainroad = st.selectbox("Main Road", ["Yes", "No"])
    guestroom = st.selectbox("Guest Room", ["Yes", "No"])
    basement = st.selectbox("Basement", ["Yes", "No"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
    airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
    prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
    furnishingstatus = st.selectbox("Furnishing", ["furnished", "semi-furnished", "unfurnished"])

# Predict button
if st.button("Predict Price"):
    # Encode inputs
    mainroad = 1.0 if mainroad == "Yes" else 0.0
    guestroom = 1.0 if guestroom == "Yes" else 0.0
    basement = 1.0 if basement == "Yes" else 0.0
    hotwaterheating = 1.0 if hotwaterheating == "Yes" else 0.0
    airconditioning = 1.0 if airconditioning == "Yes" else 0.0
    prefarea = 1.0 if prefarea == "Yes" else 0.0
    
    furnishing_map = {"furnished": 0.0, "semi-furnished": 1.0, "unfurnished": 2.0}
    furnishingstatus = furnishing_map[furnishingstatus]
    
    # Create input array
    input_data = np.array([[area, bedrooms, bathrooms, stories, 
                           mainroad, guestroom, basement,
                           hotwaterheating, airconditioning,
                           parking, prefarea, furnishingstatus]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    # Display result
    st.success(f"Predicted Price:  {prediction[0]:,.2f}")