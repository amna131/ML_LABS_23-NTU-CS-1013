# Name: Amna
# Reg no: 23-NTU-CS-1013
# Activity 3: COVID-19 Dataset Exploration
# COVID-19 Dataset Exploration: extract Pakistan's data, handle missing values, visualize , and detect outliers

import pandas as pd
import matplotlib.pyplot as plt

# Load the online dataset
url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"
df = pd.read_csv(url)

# Extract Pakistan data
pakistan = df[df["Country/Region"] == "Pakistan"].copy()

print("Pakistan Data:")
print(pakistan)
print()

# Check missing values
print("Missing Values:")
print(pakistan.isnull().sum())
print()

# Handle missing values
pakistan["Province/State"] = pakistan["Province/State"].fillna("Not Reported")

print("Missing Values after handling:")
print(pakistan.isnull().sum())
print()

# Convert Date column to datetime
pakistan["Date"] = pd.to_datetime(pakistan["Date"])

# Plot confirmed cases over time
plt.plot(pakistan["Date"], pakistan["Confirmed"])
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.title("COVID-19 Cases in Pakistan")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Find the day with the highest confirmed cases
highest = pakistan["Confirmed"].idxmax()

print("Day with highest confirmed cases:")
print(pakistan.loc[highest, ["Date", "Confirmed"]])
print()

# Calculate daily new cases
pakistan["Daily Cases"] = pakistan["Confirmed"].diff()

# Boxplot to detect outliers in daily cases
plt.boxplot(pakistan["Daily Cases"].dropna())
plt.ylabel("Daily Cases")
plt.title("Daily COVID-19 Cases")
plt.show()

print("Boxplot Observation:")
print("Points above the upper whisker indicate outlier days with unusually high new case counts.")