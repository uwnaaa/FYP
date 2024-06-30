import streamlit as st

col1,col2 = st.columns([1,2])
col1.title('Sum:')

with st.form('addition'):
    a = st.number_input('expenditure')
    b = st.number_input('poverty')
    submit = st.form_submit_button('add')

if submit:
    col2.title(f'{a+b:.2f}')
