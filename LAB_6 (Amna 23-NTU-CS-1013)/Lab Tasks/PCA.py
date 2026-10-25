# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 6
# Task 1: Principal Component Analysis (PCA) on Wine Dataset

# Reduce the Wine dataset to two principal components using PCA:
# standardize the data, plot the reduced data by class, view the component loadings,
# check the explained variance, and find the number of components that keep 96% of the variance

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
X = wine.data
y = wine.target
feat_names = wine.feature_names
target_names = wine.target_names

print("Orignal dataset shape:", X.shape)
print("Target shape", y.shape)
print("Feature names:", feat_names)
print("Target names:", target_names)

#Step 1: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Step 2: Reduce the data to two principal components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
print()
print ("Orignal dataset shape:", X.shape)
print ("Reduced dataset shape:", X_reduced.shape)
print()

wine_colors = ['orangered', 'darkred', 'crimson']
plt.figure(figsize=(10,6))

for class_num, color, target_name in zip(
    [0,1,2], wine_colors, target_names
):
    plt.scatter(
        X_reduced[y == class_num, 0],
        X_reduced[y == class_num, 1],
        color = color,
        label = target_name,
        alpha = 0.7
    )
plt.title('PCA Of Wine Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

print("Principal component loadings:")
print(pca.components_)
print("Shape of principal components:", pca.components_.shape)

plt.figure(figsize=(14,5))
sns.heatmap(
    pca.components_,
    cmap="viridis",
    xticklabels=feat_names,
    yticklabels=('First Component', 'Second Component')
)

plt.title('PCA Component Loadings Heatmap')
plt.xlabel("Orignal Features")
plt.ylabel("Principal Components")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Explained variance:")
print(pca.explained_variance_)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

pca_3 =PCA(n_components=0.96)
X_reduced_96 = pca_3.fit_transform(X_scaled)
print("Number of components selected:", pca_3.n_components_)
print("Total explained variance:", np.sum(pca_3.explained_variance_ratio_))
print("Reduced dataset shape", X_reduced_96.shape)

