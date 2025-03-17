import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os.path

# Load the trained model
model = joblib.load('xgmodel.pkl')

# Function to save user inputs and predictions to CSV
def save_to_csv(inputs, prediction):
    csv_filename = 'Recent_data.csv'
    # Create a DataFrame to store inputs and prediction
    columns=['Age', 'Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

    df = pd.DataFrame(inputs, columns=['Age', 'Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
    df['Calories_Burned'] = prediction

    # Check if the CSV file exists
    if not os.path.isfile(csv_filename):
        df.to_csv(csv_filename, index=False, columns=columns)  # Create new CSV file
    else:
        df.to_csv(csv_filename, mode='a', header=False, index=False)  # Append to existing CSV file

# Streamlit UI
st.title('Calories Burnt Prediction')

age = st.number_input('Age', min_value=16, max_value=100, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
height = st.number_input('Height (cm)', min_value=122, max_value=222, value=150)
weight = st.number_input('Weight (kg)', min_value=36.0, max_value=134.0, value=70.0)
duration = st.number_input('Duration of Exercise (minutes)', min_value=0.0, max_value=300.0, value=30.0)
heart_rate = st.number_input('Heart Rate (bpm)', min_value=40, max_value=200, value=100)
body_temp = st.number_input("Body temperature (Celsius)", min_value=35, max_value=42, value=37)

gender_binary = 1 if gender == 'Male' else 0
inputs = [[age, gender, height, weight, duration, 3, body_temp]]

if st.button('Predict'):
    inputs_array = np.array([[age, gender_binary, height, weight, duration, heart_rate, body_temp]])
    prediction = model.predict(inputs_array)
    st.write(f'Estimated Calories Burned: {prediction[0]:.2f} calories')
    
    # Save inputs and prediction to CSV file
    
    save_to_csv(inputs, prediction)

# Show last 5 predictions from CSV file
st.header('Recent Predictions')
if os.path.isfile('Recent_data.csv'):
    recent_predictions = pd.read_csv('Recent_data.csv')
    st.write(recent_predictions.tail())
else:
    st.write('No predictions recorded yet.')
