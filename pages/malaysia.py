import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from a CSV file into a pandas DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/population_malaysia.csv')

df

# Extracting data for plotting
states = df['state']
populations = df['population']

# Plotting
plt.figure(figsize=(10, 6))

# Bar chart for Population
plt.bar(states, populations, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('States')
plt.ylabel('Population')
plt.title('Population by State')
plt.xticks(rotation=90)  # Rotate state names for better readability

plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())

##########################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from a CSV file into a pandas DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/income_state.csv')

df
# Extracting data for plotting
states = df['state']
income = df['income_mean']
expenditure = df['expenditure_mean']
poverty = df['poverty']

# Plotting
plt.figure(figsize=(12, 18))

# Bar chart for income
plt.subplot(3, 1, 1)
plt.bar(states, income, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('States')
plt.ylabel('Income')
plt.title('Income by State')
plt.xticks(rotation=90)  # Rotate state names for better readability

# Bar chart for expenditure
plt.subplot(3, 1, 2)
plt.bar(states, expenditure, color='salmon', edgecolor='black', alpha=0.7)
plt.xlabel('States')
plt.ylabel('Expenditure')
plt.title('Expenditure by State')
plt.xticks(rotation=90)  # Rotate state names for better readability

# Bar chart for poverty
plt.subplot(3, 1, 3)
plt.bar(states, poverty, color='lightgreen', edgecolor='black', alpha=0.7)
plt.xlabel('States')
plt.ylabel('Poverty')
plt.title('Poverty by State')
plt.xticks(rotation=90)  # Rotate state names for better readability

plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())
