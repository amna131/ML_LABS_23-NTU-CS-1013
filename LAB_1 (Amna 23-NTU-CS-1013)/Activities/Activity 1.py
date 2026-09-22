# Name: Amna
# Reg no: 23-NTU-CS-1013
# Activity 1: Pakistani Provinces Analysis
# Analyze Pakistan's provinces: handle missing data, encode Region, visualize, and detect literacy rate outliers

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Create dataset
data = {
    'Province': ['Punjab', 'Sindh', 'Khyber Pakhtunkhwa', 'Balochistan'],
    'Population': [127, 57, None, 14],
    'Literacy Rate': [64, None, 72, 5],
    'Region': ['East', 'South', 'North', 'West']
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)
print()

# Handle missing values: drop one row, fill another with median
df = df.drop(1)
df['Population'] = df['Population'].fillna(df['Population'].median())

print("Data after handling missing values:")
print(df)
print()

# Label Encoding
label_encoder = LabelEncoder()
df['Region_Label'] = label_encoder.fit_transform(df['Region'])

print("Label Encoded Data:")
print(df)
print()

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Region'])

print("One-Hot Encoded Data:")
print(df)
print()

# Scatter plot: Population vs Literacy Rate (labeled by province)
plt.scatter(df['Population'], df['Literacy Rate'])

for i in range(len(df)):
    plt.text(df['Population'].iloc[i], df['Literacy Rate'].iloc[i], df['Province'].iloc[i])

plt.xlabel("Population (millions)")
plt.ylabel("Literacy Rate (%)")
plt.title("Population vs Literacy Rate")
plt.show()

# Detect outliers in Literacy Rate using IQR method
Q1 = df['Literacy Rate'].quantile(0.25)
Q3 = df['Literacy Rate'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Literacy Rate'] < lower) | (df['Literacy Rate'] > upper)]

print("Literacy Rate Outliers:")
if len(outliers) == 0:
    print("No province is an outlier in literacy rate.")
else:
    print(outliers[['Province', 'Literacy Rate']])