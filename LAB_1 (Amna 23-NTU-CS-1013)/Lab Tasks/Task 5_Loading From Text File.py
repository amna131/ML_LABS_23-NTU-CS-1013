# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task 5
# Load a text file into a DataFrame
import pandas as pd

# custom delimiter
df_text = pd.read_csv('data.txt', delimiter='\t')
print("Text File Data:")
print(df_text.head())