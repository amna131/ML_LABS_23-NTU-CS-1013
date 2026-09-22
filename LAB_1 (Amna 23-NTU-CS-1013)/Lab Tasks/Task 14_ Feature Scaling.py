# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task 14
# Scale numerical data using StandardScaler and MinMaxScaler from sklearn

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.DataFrame({'Age': [22, 25, 47, 35, 60], 'Income': [25000, 32000, 95000, 60000, 150000]})

# Standardization
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)

# Normalization
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)

print("Original Data:\n", data)
print("\nStandardized Data:\n", data_standard)
print("\nMin-Max Scaled Data:\n", data_minmax)