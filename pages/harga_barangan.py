import pandas as pd
df_ayam = pd.read_csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/df_ayam.csv')
df_ayam.drop(columns=['premise', 'address', 'state'])





