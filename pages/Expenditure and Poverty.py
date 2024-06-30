import streamlit as st
import pandas as pd
import numpy as np
import joblib

col1,col2 = st.columns([1,2])
col1.title('Poverty Analysis:')

with st.form('addition'):
    a = st.number_input('Income')
    b = st.number_input('Expenditure')
    submit = st.form_submit_button('Submit')

if submit:
    col2.title(f'{a+b:.2f}')

# Load your model from the pkl file
model = joblib.load('path_to_your_model.pkl')

# Define function to predict poverty rate and cluster
def predict_poverty_rate(income, expenditure):
    # Example: Use your model to predict poverty rate range and cluster
    # Replace this with your actual prediction logic
    poverty_rate = np.random.randint(0, 100)  # Example random prediction
    cluster = np.random.choice(['Cluster A', 'Cluster B', 'Cluster C'])  # Example random cluster

    return poverty_rate, cluster

# Streamlit app
def main():
    st.title('Poverty Rate Predictor')

    # Input form
    income = st.number_input('Enter Income', min_value=0.0, step=1000.0)
    expenditure = st.number_input('Enter Expenditure', min_value=0.0, step=100.0)

    # Submit button
    if st.button('Predict'):
        poverty_rate, cluster = predict_poverty_rate(income, expenditure)

        # Display results
        st.write(f'Predicted Poverty Rate: {poverty_rate}%')
        st.write(f'Cluster: {cluster}')

if __name__ == '__main__':
    main()

