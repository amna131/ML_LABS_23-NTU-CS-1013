# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task 8
# Handling Missing Values in a DataFrame
# Methods: Drop rows, Drop columns, Fill with median

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create a sample dataset containing missing values (None)
data = {
    'A': [1, 2, None, 4, 5],
    'B': [6, None, 8, 9, 10],
    'C': [11, 12, 13, None, 15]
}
df = pd.DataFrame(data)

# Original DataFrame
print("Original DataFrame:")
print(df)

# Handling Missing Values

# Method 1: Drop all rows that contain any missing values
df_dropna_rows = df.dropna(axis=0)

# Method 2: Drop all columns that contain any missing values
df_dropna_columns = df.dropna(axis=1)

# Method 3: Fill missing values with the median of each column
df_fill_median = df.fillna(df.median())

#After handling missing values
print("\nDataFrame after dropping rows with missing values:")
print(df_dropna_rows)

print("\nDataFrame after dropping columns with missing values:")
print(df_dropna_columns)

print("\nDataFrame after filling missing values with the median:")
print(df_fill_median)