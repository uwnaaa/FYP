import streamlit as st

col1,col2 = st.columns([1,2])
col1.title('Poverty Analysis:')

with st.form('addition'):
    a = st.number_input('Income')
    b = st.number_input('Expenditure')
    submit = st.form_submit_button('Submit')

if submit:
    col2.title(f'{a+b:.2f}')
