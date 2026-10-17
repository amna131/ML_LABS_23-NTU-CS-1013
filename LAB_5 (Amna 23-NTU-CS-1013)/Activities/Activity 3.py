# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 5
# Activity 3: Customer Segmentation using K-Means and DBSCAN

# Segment mall customers using K-Means and DBSCAN on Annual Income and Spending Score:
# compare results before and after scaling, find the best k using the Elbow Method,
# interpret each cluster, and study the effect of adding a random feature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

#1. ---Load and display dataset---
data = pd.read_csv("Mall_Customers.csv")
print(data.head())
print("Dataset shape:", data.shape)
print()

# Features used for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

#2. ---Apply K-Means and DBSCAN (without scaling)---
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

dbscan = DBSCAN(eps=5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

print("K-Means cluster counts:\n", pd.Series(kmeans_labels).value_counts())
print()
print("DBSCAN cluster counts (-1 = noise):\n", pd.Series(dbscan_labels).value_counts())
print()

#3. ---Scale features using StandardScaler and re-run K-Means---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=5, random_state=42)
kmeans_scaled_labels = kmeans_scaled.fit_predict(X_scaled)

print("K-Means (scaled) cluster counts:\n", pd.Series(kmeans_scaled_labels).value_counts())
print()

# Plot all three results using Annual Income vs Spending Score
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('K-Means (Without Scaling)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50)
plt.title('DBSCAN (Without Scaling)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_scaled_labels, cmap='viridis', s=50)
plt.title('K-Means (With Scaling)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.tight_layout()
plt.show()

#4. ---Compare results---
# Rows = clusters without scaling, columns = clusters with scaling
print("Comparison of K-Means clusters (rows: unscaled, columns: scaled):")
print(pd.crosstab(kmeans_labels, kmeans_scaled_labels))
print()
# Did scaling change the clusters?
# No. The K-Means clusters are the same before and after scaling,
# and every customer stayed in the same cluster.
# Annual Income (15-137) and Spending Score (1-99) are
# already on similar scales, so StandardScaler had no effect on the result.

#5. ---Elbow Method to find the best k---
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

#6. ---Interpret each cluster---
data['Cluster'] = kmeans_scaled_labels
print("Average Annual Income and Spending Score of each cluster:")
print(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
print()
# Briefly interpret each cluster.
# Cluster interpretation:
# Cluster 0 (81 customers): average income and average spending (average customers)
# Cluster 1 (39 customers): high income, high spending (best customers)
# Cluster 2 (22 customers): low income, high spending
# Cluster 3 (35 customers): high income, low spending (careful savers)
# Cluster 4 (23 customers): low income, low spending

#7. ---Add a fake feature "LuckyNumber" (random 0-100)---
np.random.seed(42)
data['LuckyNumber'] = np.random.randint(0, 101, size=len(data))

X_lucky = data[['Annual Income (k$)', 'Spending Score (1-100)', 'LuckyNumber']].values
X_lucky_scaled = StandardScaler().fit_transform(X_lucky)

kmeans_lucky = KMeans(n_clusters=5, random_state=42)
lucky_labels = kmeans_lucky.fit_predict(X_lucky_scaled)

print("Comparison of clusters (rows: without LuckyNumber, columns: with LuckyNumber):")
print(pd.crosstab(kmeans_scaled_labels, lucky_labels))
print()

# Plot clusters with and without LuckyNumber
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_scaled_labels, cmap='viridis', s=50)
plt.title('K-Means Without LuckyNumber')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=lucky_labels, cmap='viridis', s=50)
plt.title('K-Means With LuckyNumber')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.tight_layout()
plt.show()

#LuckyNumber is random and has no meaning,
# but K-Means treats it as a real feature.
# It adds noise to the distance calculation,
# so customers with similar income and spending got split into different clusters (about 40% changed cluster)
# and the clusters became mixed. Irrelevant features should be removed before clustering.