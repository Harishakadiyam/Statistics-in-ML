import streamlit as st
import pickle
import numpy as np

# Load the dataset
model  = pickle.load(open(r'C:\Users\Admin\AVScode\Investment_model_pkl',"rb"))

# Set the title of the Streamlit app
st.title("INVESTMENT")

# Add a brief description
st.write("This app predicts the right investment using a Multiple linear regression model")

# Add input widget for user to enter Actual Vs Predict
digital = st.sidebar.number_input(" Digital Marketing Investment ($)", min_value=0.0, step=1000.0)
promotion = st.sidebar.number_input(" Promotion Investment ($)", min_value=0.0, step=1000.0)
research = st.sidebar.number_input(" Research Investment ($)", min_value=0.0, step=1000.0)

state = st.sidebar.selectbox(" Select State", ("Hyderabad", "Bangalore", "Chennai"))

# Convert state into one-hot encoding (like training)
state_map = {
    "Hyderabad": [1, 0, 0],
    "Bangalore": [0, 1, 0],
    "Chennai": [0, 0, 1]
}
    
# When the button is clicked, make predictions
if st.sidebar.button(" Predict Profit"):
    # Create input array (must match training format)
    input_data = np.array([[digital]])
    prediction = model.predict(input_data)
   
    # Display the result
    st.success(f" Predicted Profit: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model of Investment.built by Harisha")

import os
os.getcwd()