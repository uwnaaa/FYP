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
st.header('Kmean', divider='rainbow')
import matplotlib.pyplot as plt
import seaborn as sns

# Plot clusters
sns.pairplot(df, hue='cluster', vars=columns_to_cluster)
plt.show()
plt.tight_layout()
st.pyplot(plt.gcf())
########################################################
st.header('Testing and Test', divider='rainbow')
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
st.write("X_train shape:", X_train.shape)
st.write("X_test shape:", X_test.shape)
st.write("y_train shape:", y_train.shape)
st.write("y_test shape:", y_test.shape)
##################################################################
st.header('Training and Testing Norm Shape', divider='rainbow')
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
st.write("X_train_norm shape:", X_train_norm.shape)
st.write("X_test_norm shape:", X_test_norm.shape)
#############################################################
st.header('SSE Kmeans', divider='rainbow')
import streamlit as st
import pandas as pd
import numpy as np
sse_org =kmeans.inertia_
st.write("SSE of Given data =" , sse_org)
###############################################################
st.header('Silhouette', divider='rainbow')
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# Sample data
X = np.random.rand(100, 5)  # Replace with your actual data

# Normalize the data
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_norm)

# Check the shapes of the arrays
print("Shape of X_train_norm:", X_train_norm.shape)  # Should be (n_samples, n_features)
print("Shape of kmeans.labels_:", kmeans.labels_.shape)  # Should be (n_samples,)

# Calculate silhouette score
silhouette_avg = silhouette_score(X_train_norm, kmeans.labels_)

# Display the silhouette score in Streamlit
st.write("Silhouette score of Given data's cluster result =", silhouette_avg)

#####################################################################################################
st.header('DBSCAN', divider='rainbow')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Load the parquet file
url = 'https://storage.dosm.gov.my/hies/hies_district.parquet'
df = pd.read_parquet(url)

# Inspect the DataFrame
print(df.head())
print(df.columns)

# Assuming the dataset has columns 'income_mean', 'expenditure_mean', and 'poverty'
income_col = 'income_mean'       # Replace with the actual column name for income
expenditure_col = 'expenditure_mean'  # Replace with the actual column name for expenditure
poverty_col = 'poverty'     # Replace with the actual column name for poverty

# Check for NaNs
print(df.isnull().sum())

# Step 2: Preprocess the data
X = df[[income_col, expenditure_col]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply DBSCAN Clustering
model = DBSCAN(eps=0.5, min_samples=5)
yhat = model.fit_predict(X_scaled)

# Add cluster labels to the dataframe
df['cluster'] = yhat

# Step 4: Analyze Clusters
# Calculate mean income, mean expenditure, Gini coefficient, poverty rate, and SSE for each cluster
results = []
clusters = np.unique(yhat)
for cluster in clusters:
    cluster_data = df[df['cluster'] == cluster]
    income_mean = cluster_data[income_col].mean()
    expenditure_mean = cluster_data[expenditure_col].mean()

    # Calculate Gini coefficient
    income_vals = cluster_data[income_col].values
    income_sorted = np.sort(income_vals)
    index = np.arange(1, len(income_vals) + 1)
    gini = (2 * np.sum(index * income_sorted)) / (len(income_vals) * np.sum(income_sorted)) - (len(income_vals) + 1) / len(income_vals)

    poverty_rate = cluster_data[poverty_col].mean() * 100  # Assuming poverty is a binary column

    # Calculate SSE
    cluster_points = X_scaled[df['cluster'] == cluster]
    if len(cluster_points) > 1:
        centroid = np.mean(cluster_points, axis=0)
        sse = np.sum(np.square(cdist(cluster_points, [centroid])))
    else:
        sse = 0  # SSE is zero if there's only one point in the cluster

    results.append({
        'cluster': cluster,
        'income_mean': income_mean,
        'expenditure_mean': expenditure_mean,
        'gini': gini,
        'poverty_rate': poverty_rate,
        'sse': sse
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Step 5: Visualize Clusters
plt.figure(figsize=(10, 7))
for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    plt.scatter(X[row_ix, 0], X[row_ix, 1], label=f'Cluster {cluster}')
plt.legend()
plt.xlabel('Income')
plt.ylabel('Expenditure')
plt.title('DBSCAN Clusters')
plt.show()
plt.tight_layout()
st.pyplot(plt.gcf())
# Print results
print(results_df)
st.write("sse=", np_sum)
#######################################################################
st.header('Hierarchical Clustering', divider='rainbow')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Load the parquet file
url = 'https://storage.dosm.gov.my/hies/hies_district.parquet'
df = pd.read_parquet(url)

# Inspect the DataFrame
print(df.head())
print(df.columns)

# Assuming the dataset has columns 'income_mean', 'expenditure_mean', and 'poverty'
income_col = 'income_mean'       # Replace with the actual column name for income
expenditure_col = 'expenditure_mean'  # Replace with the actual column name for expenditure
poverty_col = 'poverty'     # Replace with the actual column name for poverty

# Check for NaNs
print(df.isnull().sum())

# Step 2: Preprocess the data
X = df[[income_col, expenditure_col]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply DBSCAN Clustering
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
yhat_dbscan = dbscan_model.fit_predict(X_scaled)

# Apply Agglomerative Clustering
agglomerative_model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
yhat_agg = agglomerative_model.fit_predict(X_scaled)

# Add cluster labels to the dataframe
df['dbscan_cluster'] = yhat_dbscan
df['agg_cluster'] = yhat_agg

# Step 4: Analyze Clusters
# Calculate mean income, mean expenditure, Gini coefficient, poverty rate, and SSE for each cluster
results = []
clusters = np.unique(yhat_agg)
for cluster in clusters:
    cluster_data = df[df['agg_cluster'] == cluster]
    income_mean = cluster_data[income_col].mean()
    expenditure_mean = cluster_data[expenditure_col].mean()

    # Calculate Gini coefficient
    income_vals = cluster_data[income_col].values
    income_sorted = np.sort(income_vals)
    index = np.arange(1, len(income_vals) + 1)
    gini = (2 * np.sum(index * income_sorted)) / (len(income_vals) * np.sum(income_sorted)) - (len(income_vals) + 1) / len(income_vals)

    poverty_rate = cluster_data[poverty_col].mean() * 100  # Assuming poverty is a binary column

    # Calculate SSE
    cluster_points = X_scaled[df['agg_cluster'] == cluster]
    if len(cluster_points) > 1:
        centroid = np.mean(cluster_points, axis=0)
        sse = np.sum(np.square(cdist(cluster_points, [centroid])))
    else:
        sse = 0  # SSE is zero if there's only one point in the cluster

    results.append({
        'cluster': cluster,
        'income_mean': income_mean,
        'expenditure_mean': expenditure_mean,
        'gini': gini,
        'poverty_rate': poverty_rate,
        'sse': sse
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Step 5: Visualize Clusters
plt.figure(figsize=(10, 7))
for cluster in clusters:
    row_ix = np.where(yhat_agg == cluster)
    plt.scatter(X[row_ix, 0], X[row_ix, 1], label=f'Cluster {cluster}')
plt.legend()
plt.xlabel('Income')
plt.ylabel('Expenditure')
plt.title('Agglomerative Clusters')
plt.show()
plt.tight_layout()
st.pyplot(plt.gcf())
# Print results
print(results_df)
