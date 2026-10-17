# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab Task 3 Part 2: K-Means Clustering (Confusion Matrix)

# Cluster the mushroom dataset using K-Means (2 clusters), compare the clusters
# with the true labels using accuracy, confusion matrix and classification report

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#1. Load the dataset
mushroom_data = pd.read_csv("mushroomss.csv")

# Handle missing values
mushroom_df = mushroom_data.dropna()

# Separate features and target
# Note: the dataset has no 'class' column, so 'bruises' is used as the label instead
mushroom_features = mushroom_df.drop('bruises', axis=1)
mushroom_labels = mushroom_df['bruises']

#2. Encode categorical features
encoder = OrdinalEncoder()
mushroom_prepared = encoder.fit_transform(mushroom_features)
mushroom_prep_df = pd.DataFrame(mushroom_prepared, columns=mushroom_features.columns)

#3. ---KMeans Clustering---
# Create and fit KMeans model
k_clust = KMeans(n_clusters=2, random_state=42)
k_clust.fit(mushroom_prep_df)

# Get results
k_labels = k_clust.labels_

# Map true labels to numerical values
map_dict = {'f': 0, 't': 1}
true_labels = mushroom_labels.map(map_dict)

#4. Calculate accuracy
accuracy = accuracy_score(true_labels, k_labels)
print(f"KMeans Accuracy: {accuracy:.4f}")

#5. Plot confusion matrix
plt.figure(figsize=(8, 6))

cm = confusion_matrix(true_labels, k_labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cluster 0', 'Cluster 1'],
            yticklabels=['No Bruises', 'Bruises'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted Clusters')
plt.ylabel('True Labels')
plt.show()

#6. Classification report
print("\nClassification Report:")
print(classification_report(true_labels, k_labels, target_names=['No Bruises', 'Bruises']))