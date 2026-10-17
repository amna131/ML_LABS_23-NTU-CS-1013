# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab Task 3 Part 1: K-Means Clustering (Data Exploration)

# Load the mushroom dataset, check missing values and visualize
# the distribution of key features using bruises as the grouping column

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#1. Load the dataset
mushroom_data = pd.read_csv("mushroomss.csv")

# Display first few rows
print("Dataset Preview:")
print(mushroom_data.head())

#2. Check for missing values
print("\nMissing Values in Each Column:")
print(mushroom_data.isnull().sum())

print(f"Dataset shape: {mushroom_data.shape}")

# Note: the dataset has no 'class' column, so 'bruises' is used instead
print(f"Bruises distribution:\n{mushroom_data['bruises'].value_counts()}")

#3. Visualize the data
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.countplot(data=mushroom_data, x='bruises')
plt.title('Bruises Distribution')

plt.subplot(2, 3, 2)
sns.countplot(data=mushroom_data, x='cap-shape', hue='bruises')
plt.title('Cap Shape vs Bruises')

plt.subplot(2, 3, 3)
sns.countplot(data=mushroom_data, x='cap-color', hue='bruises')
plt.title('Cap Color vs Bruises')

plt.subplot(2, 3, 4)
sns.countplot(data=mushroom_data, x='population')
plt.title('Population Distribution')

plt.subplot(2, 3, 5)
sns.countplot(data=mushroom_data, x='habitat')
plt.title('Habitat Distribution')

plt.subplot(2, 3, 6)
sns.countplot(data=mushroom_data, x='odor', hue='bruises')
plt.title('Odor vs Bruises')

plt.tight_layout(pad=3)
plt.show()