#create environment
#python -m 

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
model = pickle.load(open('lr_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb')) 
# input_scaled = model.transform(input_features)
# prediction = model.predict(input_scaled)
# Title
st.title("House Price Prediction App")

# Input fields
square_footage = st.number_input('Square Footage', min_value=300, max_value=10000, value=1500)
num_bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
num_bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
year_built = st.number_input('Year Built', min_value=1900, max_value=2025, value=2000)
lot_size = st.number_input('Lot Size', min_value=0.1, max_value=10.0, value=2.5)
garage_size = st.number_input('Garage Size', min_value=0, max_value=5, value=1)
neighborhood_quality = st.number_input('Neighborhood Quality (1-10)', min_value=1, max_value=10, value=5)

# Create DataFrame
input_features = pd.DataFrame({
    'Square_Footage': [square_footage],
    'Num_Bedrooms': [num_bedrooms],
    'Num_Bathrooms': [num_bathrooms],
    'Year_Built': [year_built],
    'Lot_Size': [lot_size],
    'Garage_Size': [garage_size],
    'Neighborhood_Quality': [neighborhood_quality]
})
if st.button('Predict'):
    
    # Apply SAME scaling as training
    input_scaled = scaler.transform(input_features)
    
    # Predict (log scale)
    prediction_log = model.predict(input_scaled)
    
    # Convert back to original price
    prediction = np.exp(prediction_log[0])
    
    # Round result
    output = round(prediction, 2)
    
    st.success(f'Predicted House Price:  ₹{output:,.2f}')




