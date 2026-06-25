# Question 11: Implement K-means clustering and justify your answers.
# Aim: Cluster the encoded dataset into 2 groups and inspect assignments.

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("credit-g_csv.csv")

# Convert categorical attributes into numeric values
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(data)

# Get cluster labels
labels = kmeans.labels_

# Add cluster column
data['Cluster'] = labels

# Display cluster counts
print("Cluster Assignments:")
print(data['Cluster'].value_counts())
