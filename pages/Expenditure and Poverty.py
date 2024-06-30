import streamlit as st
import pandas as pd
import numpy as np
import joblib

col1,col2 = st.columns([1,2])
col1.title(''Poverty Rate and Cluster Predictor:')

with st.form('addition'):
    a = st.number_input('Income')
    b = st.number_input('Expenditure')
    submit = st.form_submit_button('Submit')

if submit:
    col2.title(f'{a+b:.2f}')

# Load your model from the pkl file
model = joblib.load('pages/clustering_model.pkl')

# Function to predict poverty rate and cluster
def predict_poverty_rate_and_cluster(income, expenditure):
    # Assuming model.predict() returns a tuple (poverty_rate, cluster)
    poverty_rate, cluster = model.predict([[income, expenditure]])
    return poverty_rate[0], cluster[0]

 # Submit button
    if st.button('Predict'):
        poverty_rate, cluster = predict_poverty_rate_and_cluster(income, expenditure)

        # Display results
        st.write(f'Predicted Poverty Rate: {poverty_rate}%')
        st.write(f'Cluster: {cluster}')

if __name__ == '__main__':
    main()

