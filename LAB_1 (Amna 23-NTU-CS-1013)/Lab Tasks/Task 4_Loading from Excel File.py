# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task 4
# Load a dataset from an Excel file into a DataFrame
import pandas as pd

# Load Excel file
df_excel = pd.read_excel('extra.xlsx')
print("Excel Data:")
print(df_excel.head())