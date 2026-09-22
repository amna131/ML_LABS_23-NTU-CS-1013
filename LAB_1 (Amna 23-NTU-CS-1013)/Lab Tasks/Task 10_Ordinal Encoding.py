# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task 10
# Ordinal Encoding Example
# Using OrdinalEncoder to encode ordered categorical data

from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

data = pd.DataFrame({'Education': ['High School', 'Bachelors', 'Masters', 'PhD', 'Bachelors']})
order = ['High School', 'Bachelors', 'Masters', 'PhD']

oe = OrdinalEncoder(categories=[order])
data['Education_Encoded'] = oe.fit_transform(data[['Education']])

print(data)