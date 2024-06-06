import pandas as pd
df = pd_read.csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/df_ayam.csv')
df_ayam = df_ayam.merge(location, on = 'premise_code', how = 'left')
df_ayam





