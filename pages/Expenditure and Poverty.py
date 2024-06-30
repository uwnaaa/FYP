import streamlit as st
import pandas as pd
import numpy as np
import joblib

col1,col2 = st.columns([1,2])
col1.title('Poverty Rate Analysis')

with st.form('addition'):
    Income = st.number_input('Income')
    Expenditure = st.number_input('Expenditure')
    Submit = st.form_submit_button('Submit')

if submit:
    poverty_rate, cluster = predict_poverty_rate_and_cluster(income, expenditure)

    # Display results
    st.write(f'Predicted Poverty Rate:' {poverty_rate}%)
    st.write(f'Cluster:' {cluster})

# Load your model from the pkl file
model = joblib.load('pages/clustering_model.pkl')

# Function to predict poverty rate and cluster
def predict_poverty_rate_and_cluster(income, expenditure):
    # Assuming model.predict() returns a tuple (poverty_rate, cluster)
    poverty_rate, cluster = model.predict([[income, expenditure]])
    return poverty_rate[0], cluster[0]





