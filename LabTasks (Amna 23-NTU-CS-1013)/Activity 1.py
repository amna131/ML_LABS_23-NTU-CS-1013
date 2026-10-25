# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 6
# Activity 1: PCA on Fashion MNIST

# Apply PCA on the Fashion MNIST dataset using three component-selection approaches
# (fixed number of components, percentage of variance, all components) and compare
# their effects on dimensionality reduction and information retention

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#1. ---Load and explore dataset---
data = pd.read_csv("fashion-mnist_train.csv")

print("Dataset shape:", data.shape)
print("\nFirst 10 rows:")
print(data.head(10))

print("\nClass distribution:")
print(data['label'].value_counts().sort_index())

#2. ---Separate features and target---
X = data.drop('label', axis=1)
y = data['label']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

#3. ---Preprocess: normalize and scale the pixel values---
# Normalize the pixel values from 0-255 to 0-1
X_normalized = X / 255.0

# Standardize each pixel (mean 0, standard deviation 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_normalized)

# Effect of scaling: compare the components needed for 95% variance
pca_normalized = PCA(n_components=0.95)
pca_normalized.fit(X_normalized)

pca_scaled = PCA(n_components=0.95)
pca_scaled.fit(X_scaled)

print("\nComponents needed for 95% variance (normalized only):", pca_normalized.n_components_)
print("Components needed for 95% variance (after StandardScaler):", pca_scaled.n_components_)

#4. ---Method 1: Fixed number of components---
pca_1 = PCA(n_components=50)
X_pca_1 = pca_1.fit_transform(X_scaled)

print("\nMethod 1: n_components=50")
print("Original number of features:", X_scaled.shape[1])
print("Number of retained components:", pca_1.n_components_)
print("Cumulative explained variance:", np.sum(pca_1.explained_variance_ratio_))

#5. ---Method 2: Percentage of variance---
pca_2 = PCA(n_components=0.95)
X_pca_2 = pca_2.fit_transform(X_scaled)

print("\nMethod 2: n_components=0.95")
print("Original number of features:", X_scaled.shape[1])
print("Number of retained components:", pca_2.n_components_)
print("Cumulative explained variance:", np.sum(pca_2.explained_variance_ratio_))

#6. ---Method 3: All components---
pca_3 = PCA(n_components=None)
X_pca_3 = pca_3.fit_transform(X_scaled)

print("\nMethod 3: n_components=None")
print("Original number of features:", X_scaled.shape[1])
print("Number of retained components:", pca_3.n_components_)
print("Cumulative explained variance:", np.sum(pca_3.explained_variance_ratio_))

#7. ---Plot cumulative explained variance---
cumsum = np.cumsum(pca_3.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum) + 1), cumsum)
plt.axhline(y=0.95, color='red', linestyle='--', label='95% variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

#8. ---Visualize the first two principal components---
plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    X_pca_3[:, 0],
    X_pca_3[:, 1],
    c=y,
    cmap='jet',
    s=5,
    alpha=0.6
)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization of Fashion MNIST')
plt.colorbar(scatter, label='Clothing Class')
plt.grid(alpha=0.2)
plt.show()

#9. ---Compare all three approaches---
original_features = X_scaled.shape[1]

comparison = pd.DataFrame({
    'Method': ['Method 1 (n_components=50)', 'Method 2 (n_components=0.95)', 'Method 3 (n_components=None)'],
    'Original Features': [original_features, original_features, original_features],
    'Retained Components': [pca_1.n_components_, pca_2.n_components_, pca_3.n_components_],
    'Cumulative Variance (%)': [
        round(np.sum(pca_1.explained_variance_ratio_) * 100, 2),
        round(np.sum(pca_2.explained_variance_ratio_) * 100, 2),
        round(np.sum(pca_3.explained_variance_ratio_) * 100, 2)
    ],
    'Dimensionality Reduction (%)': [
        round((1 - pca_1.n_components_ / original_features) * 100, 2),
        round((1 - pca_2.n_components_ / original_features) * 100, 2),
        round((1 - pca_3.n_components_ / original_features) * 100, 2)
    ]
})

print("\nComparison of the three approaches:")
print(comparison.to_string(index=False))

# Method 1 gives the greatest dimensionality reduction (only 50 components).
# Method 3 preserves the most information (100% variance), but gives no reduction.
# Method 2 is a balance: it keeps at least 95% of the variance with fewer components.