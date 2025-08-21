import streamlit as st
import pickle
import pandas as pd
import numpy as np

#load the save model
model = pickle.load(open(r'C:\Users\Admin\AVScode\house_price_model_pkl',"rb"))

# Set the title of the Streamlit app
st.title("House Prediction App")

# Add a brief description
st.write("This app predicts the Price based on Square Feet using a simple linear regression model")

# Add input widget for user to enter square feet
Price_Sqft = st.number_input("Enter sqft:", min_value=0.0, max_value=5000.0, value=1.0, step=0.0)

# When the button is clicked, make predictions
if st.button("Price Prediction"):
    # Make a prediction using the trained model
    Sqft_input = np.array([[Price_Sqft]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(Sqft_input)

# Display the result
    st.success(f"The predicted House Price for sqft is: ${prediction[0]:,.2f}")

   
# Display information about the model
st.write("The model was trained using a dataset of House and price of Sqft.built model by Harisha")

