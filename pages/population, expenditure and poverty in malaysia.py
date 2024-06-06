import streamlit as st
st.set_page_config(
   page_title="Population in Malaysia",)
st.header('Population in Malaysia', divider='rainbow')

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
st.header('Income, Expenditure and Poverty in Malaysia', divider='rainbow')
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
tab1, tab2, tab3 = st.tabs(["Income by State", "Expenditure by State", "Poverty by State"])
with tab1:
   st.header("Income by State")
   plt.figure(figsize=(12, 18))
   # Bar chart for income
   plt.subplot()
   plt.bar(states, income, color='skyblue', edgecolor='black', alpha=0.7)
   plt.xlabel('States')
   plt.ylabel('Income')
   plt.title('Income by State')
   plt.xticks(rotation=90)  # Rotate state names for better readability
   plt.tight_layout()
   plt.show()
   st.pyplot(plt.gcf())
  
with tab2:
   st.header("Expenditure by State")
   plt.figure(figsize=(12, 18))

   # Bar chart for expenditure
   plt.subplot()
   plt.bar(states, expenditure, color='salmon', edgecolor='black', alpha=0.7)
   plt.xlabel('States')
   plt.ylabel('Expenditure')
   plt.title('Expenditure by State')
   plt.xticks(rotation=90)  # Rotate state names for better readability
   plt.tight_layout()
   plt.show()
   st.pyplot(plt.gcf())

with tab3:
   st.header("Poverty by State")
   plt.figure(figsize=(12, 18))

   # Bar chart for poverty
   plt.subplot()
   plt.bar(states, poverty, color='lightgreen', edgecolor='black', alpha=0.7)
   plt.xlabel('States')
   plt.ylabel('Poverty')
   plt.title('Poverty by State')
   plt.xticks(rotation=90)  # Rotate state names for better readability
   plt.tight_layout()
   plt.show()
   st.pyplot(plt.gcf())



############################################
st.header('Income and Expenditure by Selangor and Perlis', divider='rainbow')
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have data in a dictionary format like this:
state_data = {
    'Selangor': {'income_median': 9983, 'expenditure_mean': 6770, 'poverty_rate': 1.5},
    'Perlis': {'income_median': 4713, 'expenditure_mean': 3834, 'poverty_rate': 4.0}
}

# Extracting data for plotting
states = list(state_data.keys())
income_medians = [state_data[state]['income_median'] for state in states]
expenditure_means = [state_data[state]['expenditure_mean'] for state in states]

# Setting the positions and width for the bars
bar_width = 0.2
indices = np.arange(len(states))

plt.figure(figsize=(10, 7))

# Income Median
plt.bar(indices - bar_width, income_medians, bar_width, label='Income Median', color='skyblue', edgecolor='black')

# Expenditure Mean
plt.bar(indices, expenditure_means, bar_width, label='Expenditure Mean', color='salmon', edgecolor='black')

# Adding labels
plt.xlabel('States')
plt.ylabel('Values')
plt.title('Income Median, Expenditure Mean')
plt.xticks(indices, states)
plt.legend()

plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())


######################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have data in a dictionary format like this:
state_data = {
    'Selangor': {'poverty_rate': 1.5},
    'Perlis': {'poverty_rate': 4.0},
}

# Extracting data for plotting
states = list(state_data.keys())
poverty_rates = [state_data[state]['poverty_rate'] for state in states]

# Plotting
plt.figure(figsize=(8, 6))

# Bar chart for Poverty Rate
plt.bar(states, poverty_rates, color='red', edgecolor='black', alpha=0.7)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate by State')

plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())
