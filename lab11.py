# Question 11: Implement K-means clustering and justify your answers.
# Aim: Cluster the encoded dataset into 2 groups and inspect assignments.

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"C:\Users\cheth\OneDrive\Documents\6th sem\GermanCredit.csv")

# Convert categorical columns to numeric
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Create K-Means model (2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# Train the model
kmeans.fit(data)

# Add cluster labels
data["Cluster"] = kmeans.labels_

# Display cluster assignments
print("Cluster Assignments:")
print(data["Cluster"].value_counts())

# Display first 10 records with cluster labels
print("\nFirst 10 Records:")
print(data[["Cluster"]].head(10))


# What is the Question Asking?
# The question asks you to:
# - Apply the K-Means clustering algorithm.
# - Divide the dataset into 2 clusters.
# - Show how many records belong to each cluster.
# - Explain the result.

# Viva Questions

# 1. What is K-Means?
# Answer: K-Means is an unsupervised machine learning algorithm used to
# group similar data into clusters.

# 2. Is K-Means supervised or unsupervised?
# Answer: Unsupervised.

# 3. What is a cluster?
# Answer: A group of similar data points.

# 4. What does K represent?
# Answer: The number of clusters.

# 5. Why did you choose K = 2?
# Answer: Because the experiment requires dividing the dataset into two clusters.

# 6. What is a centroid?
# Answer: The center point of a cluster.

# 7. How does K-Means work?
# Answer:
# - Choose K centroids.
# - Assign each data point to the nearest centroid.
# - Recalculate the centroids.
# - Repeat until the centroids stop changing.

# 8. Why did you use Label Encoding?
# Answer: K-Means works only with numerical data, so categorical values
# must be converted into numbers.

# 9. Can K-Means classify data?
# Answer: No. K-Means performs clustering, not classification.

# 10. What is the output of K-Means?
# Answer: It assigns each record to a cluster (Cluster 0, Cluster 1, etc.).

# 11. Which WEKA algorithm did you use?
# Answer: SimpleKMeans.

# 12. What is the conclusion?
# Answer: K-Means successfully grouped similar records into two clusters
# without using class labels.
