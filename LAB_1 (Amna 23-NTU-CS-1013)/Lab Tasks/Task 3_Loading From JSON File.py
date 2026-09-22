# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task 3
# Load a dataset from a local JSON file and from an online JSON URL

import pandas as pd

# Load JSON from local file
df_json = pd.read_json('attendance.json')
print("JSON Data:")
print(df_json.head())

# Load JSON from online source
json_url = 'https://jsonplaceholder.typicode.com/users'
df_online_json = pd.read_json(json_url)
print("\nOnline JSON Data:")
print(df_online_json.head())