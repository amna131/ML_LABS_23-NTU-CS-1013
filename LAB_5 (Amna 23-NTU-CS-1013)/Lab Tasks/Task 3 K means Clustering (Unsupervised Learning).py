# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab Task 3 Part 3: K-Means Clustering on Mushroom Dataset (Unsupervised Learning)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv("mushroomss.csv")

print("Dataset Preview:")
print(data.head())
print("\nOriginal Shape:", data.shape)

data = data.replace(["?", " ", ""], pd.NA)
data = data.dropna(axis=1, how="all")

# Fill missing values according to column type
for column in data.columns:
    if data[column].dtype == "object" or pd.api.types.is_string_dtype(data[column]):
        mode_values = data[column].mode()
        if not mode_values.empty:
            data[column] = data[column].fillna(mode_values[0])
    else:
        data[column] = data[column].fillna(data[column].median())

# Encode categorical columns
for column in data.columns:
    if data[column].dtype == "object" or pd.api.types.is_string_dtype(data[column]):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors="coerce")
data = data.dropna(axis=1, how="all")
data = data.fillna(0)

print("\nCleaned Shape:", data.shape)

if data.shape[0] == 0 or data.shape[1] == 0:
    print("Error: No usable data remains. Check your CSV file.")
else:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)

    data["Cluster"] = clusters

    print("\nCluster Centers:\n", kmeans.cluster_centers_)
    print("\nCluster Counts:\n", data["Cluster"].value_counts())

    features = data.drop("Cluster", axis=1)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                c=data["Cluster"], cmap="viridis", s=50)
    plt.title("K-Means Clustering Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()