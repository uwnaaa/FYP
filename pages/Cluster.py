import pandas as pd

# Load the data
url = 'https://storage.dosm.gov.my/hies/hies_district.parquet'
df = pd.read_parquet(url)

####################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Select relevant columns for clustering
columns_to_cluster = ['income_mean', 'expenditure_mean', 'gini', 'poverty']
data = df[columns_to_cluster]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

######################################################################
import streamlit as st
st.header('Elbow Method', divider='rainbow')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the parquet file
url = 'https://storage.dosm.gov.my/hies/hies_district.parquet'
df = pd.read_parquet(url)

# Extract the specified columns
selected_columns = ['income_mean', 'expenditure_mean', 'gini', 'poverty']
dataset = df[selected_columns].dropna().values

# Perform KMeans clustering
WCSS = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(dataset)
    WCSS.append(kmeans.inertia_)

# Plot the WCSS
plt.plot(range(1, 20), WCSS)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
plt.tight_layout()
st.pyplot(plt.gcf())

####################################################
from sklearn.cluster import KMeans

# Define the number of clusters
k = 4

# Fit the K-Means model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)

# Predict clusters
df['cluster'] = kmeans.predict(scaled_data)

######################################################
st.header('Cluster', divider='rainbow')
import matplotlib.pyplot as plt
import seaborn as sns

# Plot clusters
sns.pairplot(df, hue='cluster', vars=columns_to_cluster)
plt.show()
plt.tight_layout()
st.pyplot(plt.gcf())

######################################################
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
url = 'https://storage.dosm.gov.my/hies/hies_district.csv'
df = pd.read_csv(url)

# Inspect columns
print(df.columns)

# Select relevant columns
X = df[['income_mean', 'gini', 'expenditure_mean', 'poverty']]  # Replace with actual column names
y = df[['income_mean', 'gini', 'expenditure_mean', 'poverty']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Verify the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

####################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Load the dataset
url = 'https://storage.dosm.gov.my/hies/hies_district.csv'
df = pd.read_csv(url)

# Inspect columns
print(df.columns)

# Select relevant columns
X = df[['income_mean', 'gini', 'expenditure_mean', 'poverty']]
y = df[['income_mean', 'gini', 'expenditure_mean', 'poverty']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Normalize the training and testing data
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

# Verify the normalization
print("X_train_norm shape:", X_train_norm.shape)
print("X_test_norm shape:", X_test_norm.shape)

##################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

# Load the dataset
url = 'https://storage.dosm.gov.my/hies/hies_district.csv'
df = pd.read_csv(url)

# Inspect columns
print(df.columns)

# Select relevant columns
X = df[['income_mean', 'gini', 'expenditure_mean', 'poverty']]
y = df[['income_mean', 'gini', 'expenditure_mean', 'poverty']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Normalize the training and testing data
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(X_train_norm)

# Predict the clusters for the training data++
train_clusters = kmeans.predict(X_train_norm)
print("Training data cluster labels:", train_clusters)

# Predict the clusters for the testing data
test_clusters = kmeans.predict(X_test_norm)
print("Testing data cluster labels:", test_clusters)

###################################################################################
st.header('SSE Kmeans', divider='rainbow')
import pandas as pd
import numpy as np
sse_org =kmeans.inertia_
print('SSE of Given data =' , sse_org)

#######################################################################
st.header('Silhouette', divider='rainbow')
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
print('Silhouette score of Given data\'s clsuter result =',silhouette_score(X_train_norm,kmeans.labels_))
