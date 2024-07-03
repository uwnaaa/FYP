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

# Assuming the dataset has columns 'income', 'expenditure', and 'poverty'
# Make sure to update these column names based on the actual column names in the DataFrame
income_col = 'income_mean'       # Replace with the actual column name for income
expenditure_col = 'expenditure_mean'  # Replace with the actual column name for expenditure
poverty_col = 'poverty'     # Replace with the actual column name for poverty

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
    gini = (2 * np.sum(np.tril(cluster_data[income_col].values[:, None] + cluster_data[income_col].values)) / (len(cluster_data[income_col]) ** 2)) - 1
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
for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    plt.scatter(X[row_ix, 0], X[row_ix, 1], label=f'Cluster {cluster}')
plt.legend()
plt.show()
plt.tight_layout()
st.pyplot(plt.gcf())

# Print results
print(results_df)

#####################################################################
st.header('SSE Kmeans', divider='rainbow')
import streamlit as st
import pandas as pd
import numpy as np
sse_org =kmeans.inertia_
st.write("SSE of Given data =" , sse_org)

#######################################################################
st.header('Silhouette', divider='rainbow')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
st.write("Silhouette score of Given data\'s clsuter result =",silhouette_score(X_train_norm,kmeans.labels_))
