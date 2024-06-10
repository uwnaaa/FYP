import pandas as pd
df_ayam = pd.read_csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/df_ayam.csv')
df_ayam = df_ayam.drop(columns=['premise', 'address', 'state'])

df_ayam

###############################
import pandas as pd
df_buah = pd.read_csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/df_buah.csv')
df_buah = df_buah.drop(columns=['premise', 'address', 'state'])

df_buah

###############################
import pandas as pd
df_sayur = pd.read_csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/df_sayur.csv')
df_sayur = df_sayur.drop(columns=['premise', 'address', 'state'])

df_sayur

###############################
import streamlit as st
st.header('Average Price for Ayam, Buah and Sayur by Premise type and Distrcit', divider='rainbow')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Extracting data for plotting
premise_type = df_ayam['premise_type']
district = df_ayam['district']
price = df_ayam['price']

# Calculate average price for each combination of district and premise type
avg_price = df_ayam.groupby(['district', 'premise_type'])['price'].mean().unstack()

# Extracting unique districts and premise types
districts = avg_price.index
premise_types = avg_price.columns

# Number of districts and premise types
n_districts = len(districts)
n_types = len(premise_types)

# Create an array for the x-axis positions
x = np.arange(n_districts) * (1.5)

# Width of the bars
width = 0.2

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each premise type
for i, premise_type in enumerate(premise_types):
    pos = x + i * width - width * (n_types / 2)
    ax.bar(pos, avg_price[premise_type], width, label=premise_type)

# Adding labels and title
ax.set_xlabel('Districts')
ax.set_ylabel('Average Price')
ax.set_title('Average Price Ayam by Premise Type and District')
ax.set_xticks(x)
ax.set_xticklabels(districts)

# Adding legend
ax.legend(title='Premise Type', loc='upper right')

plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())

# Plotting
tab1, tab2, tab3 = st.tabs(["Average Price Ayam by Premise Type and District", "Average Price Buah by Premise Type and District", "Average Price Sayur by Premise Type and District"])
with tab1:
   st.header("Average Price Ayam Premise Type and District")
   plt.figure(figsize=(12, 18))
   # Bar chart for income
   plt.subplot()
   plt.bar(states, income, color='skyblue', edgecolor='black', alpha=0.7)
   plt.xlabel('Premise Type and District')
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
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Extracting data for plotting
premise_type = df_buah['premise_type']
district = df_buah['district']
price = df_buah['price']

# Calculate average price for each combination of district and premise type
avg_price = df_buah.groupby(['district', 'premise_type'])['price'].mean().unstack()

# Extracting unique districts and premise types
districts = avg_price.index
premise_types = avg_price.columns

# Number of districts and premise types
n_districts = len(districts)
n_types = len(premise_types)

# Create an array for the x-axis positions
x = np.arange(n_districts) * (1.5)

# Width of the bars
width = 0.2

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each premise type
for i, premise_type in enumerate(premise_types):
    pos = x + i * width - width * (n_types / 2)
    ax.bar(pos, avg_price[premise_type], width, label=premise_type)

# Adding labels and title
ax.set_xlabel('Districts')
ax.set_ylabel('Average Price')
ax.set_title('Average Price Buah by Premise Type and District')
ax.set_xticks(x)
ax.set_xticklabels(districts)

# Adding legend
ax.legend(title='Premise Type', loc='upper right')

plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())


##################################################
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Extracting data for plotting
premise_type = df_sayur['premise_type']
district = df_sayur['district']
price = df_sayur['price']

# Calculate average price for each combination of district and premise type
avg_price = df_sayur.groupby(['district', 'premise_type'])['price'].mean().unstack()

# Extracting unique districts and premise types
districts = avg_price.index
premise_types = avg_price.columns

# Number of districts and premise types
n_districts = len(districts)
n_types = len(premise_types)

# Create an array for the x-axis positions
x = np.arange(n_districts) * (1.5)

# Width of the bars
width = 0.2

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each premise type
for i, premise_type in enumerate(premise_types):
    pos = x + i * width - width * (n_types / 2)
    ax.bar(pos, avg_price[premise_type], width, label=premise_type)

# Adding labels and title
ax.set_xlabel('Districts')
ax.set_ylabel('Average Price')
ax.set_title('Average Price Sayur by Premise Type and District')
ax.set_xticks(x)
ax.set_xticklabels(districts)

# Adding legend
ax.legend(title='Premise Type', loc='upper right')

plt.tight_layout()
plt.show()
st.pyplot(plt.gcf())


#######################################
import pandas as pd

URL_DATA = 'https://raw.githubusercontent.com/uwnaaa/FYP/main/selangor.csv'
df = pd.read_csv(URL_DATA)
df.loc[3:]


##############################################
import pandas as pd
selected_columns = ['income_mean', 'expenditure_mean', 'poverty', 'mean_ayam', 'mean buah', 'mean sayur']
df_selected = df[selected_columns]

# Compute the correlation matrix for the selected columns
correlation_matrix = df_selected.corr()
(correlation_matrix)






