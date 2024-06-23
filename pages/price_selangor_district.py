import streamlit as st
st.header('Price Ayam in Selangor', divider='rainbow')
import pandas as pd
df_ayam = pd.read_csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/df_ayam.csv')
df_ayam = df_ayam.drop(columns=['premise', 'address', 'state'])

df_ayam

###############################
st.header('Price Buah in Selangor', divider='rainbow')
import pandas as pd
df_buah = pd.read_csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/df_buah.csv')
df_buah = df_buah.drop(columns=['premise', 'address', 'state'])

df_buah

###############################
st.header('Price Sayur in Selangor', divider='rainbow')
import pandas as pd
df_sayur = pd.read_csv('https://raw.githubusercontent.com/uwnaaa/FYP/main/df_sayur.csv')
df_sayur = df_sayur.drop(columns=['premise', 'address', 'state'])

df_sayur

###############################
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
st.header('Average Price Ayam by Premise Type and District', divider='rainbow')

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


#######################################
st.header('Average Price Buah by Premise Type and District', divider='rainbow')
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


#######################################
st.header('Average Price Sayur by Premise Type and District', divider='rainbow')
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
st.header('Correlation Analysis', divider='rainbow')
import pandas as pd
selected_columns = ['income_mean', 'expenditure_mean', 'poverty', 'mean_ayam', 'mean buah', 'mean sayur']
df_selected = df[selected_columns]

# Compute the correlation matrix for the selected columns
correlation_matrix = df_selected.corr()
(correlation_matrix)


############################################
st.header('DF Ayam', divider='rainbow')
import pandas as pd

URL_DATA = 'https://storage.data.gov.my/pricecatcher/lookup_premise.parquet'

location = pd.read_parquet(URL_DATA)
if 'date' in location.columns: location['date'] = pd.to_datetime(location['date'])

print(location)
df_ayam.drop(columns=['premise', 'address', 'state'])
df_ayam


##########################################
st.header('DF Buah', divider='rainbow')
import pandas as pd

URL_DATA = 'https://storage.data.gov.my/pricecatcher/lookup_premise.parquet'

location = pd.read_parquet(URL_DATA)
if 'date' in location.columns: location['date'] = pd.to_datetime(location['date'])

print(location)
df_buah.drop(columns=['premise', 'address', 'state'])
df_buah


############################################
st.header('DF Sayur', divider='rainbow')
import pandas as pd

URL_DATA = 'https://storage.data.gov.my/pricecatcher/lookup_premise.parquet'

location = pd.read_parquet(URL_DATA)
if 'date' in location.columns: location['date'] = pd.to_datetime(location['date'])
df_sayur.drop(columns=['premise', 'address', 'state'])
df_sayur





