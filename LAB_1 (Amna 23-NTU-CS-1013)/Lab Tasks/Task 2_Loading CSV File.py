# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task no.2
# Load a dataset from a local CSV file and from an online CSV URL

import pandas as pd

# Load CSV from local file
df_csv = pd.read_csv('cars.csv')
print("CSV Data:")
print(df_csv.head())

# Load CSV from an online URL
online_csv_url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'
df_online_csv = pd.read_csv(online_csv_url)
print("\nOnline CSV Data:")
print(df_online_csv.head())