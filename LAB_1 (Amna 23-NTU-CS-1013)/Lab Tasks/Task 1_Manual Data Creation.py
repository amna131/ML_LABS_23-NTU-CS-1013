# Amna
# 23-NTU-CS-1013
# Task 1
# Manual Data Creation

import pandas as pd

# Create data using a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['Lahore', 'Faisalabad', 'Karachi', 'Multan'],
    'Salary': [50000, 60000, 70000, 55000]
}
df_manual = pd.DataFrame(data)
print("Manual DataFrame:")
print(df_manual)