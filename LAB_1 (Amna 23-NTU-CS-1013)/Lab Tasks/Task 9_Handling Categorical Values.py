# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task 9
# Handling Categorical Attributes
# Manual Encoding, Label Encoding, and One-Hot Encoding

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Manual Encoding Example
mapping = {'h': 1, 'u': 2, 't': 3}
features = pd.DataFrame({'Type': ['h', 'u', 't', 'h', 'u']})
features['Type'] = features['Type'].map(mapping)
print("Encoded Types (Manual Encoding):")
print(features.groupby('Type').size())
print()


# 2. Label Encoding Example
features_df = pd.DataFrame({'Regionname': ['North', 'South', 'East', 'West', 'North']})
le = LabelEncoder()
features_df['Region'] = le.fit_transform(features_df['Regionname'])

print("\nLabel Encoded Regions:")
print(features_df.value_counts())
print()


# 3. One-Hot Encoding Example
features['Method'] = ['Method1', 'Method2', 'Method1', 'Method3', 'Method2']
df_one_hot = pd.get_dummies(features['Method'])
print("\nOne Hot Encoded DataFrame:")
print(df_one_hot.value_counts())