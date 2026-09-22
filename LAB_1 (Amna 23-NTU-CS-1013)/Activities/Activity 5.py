# Name: Amna
# Reg no: 23-NTU-CS-1013
# Activity 5: Data Cleaning & Outlier Detection with Visualization
# Clean data, encode Department, detect Salary outliers, and comment on Zara's salary

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Create dataset
data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}

df = pd.DataFrame(data)


# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

print("Data after handling missing values:")
print(df)
print()

# Label Encoding
label_encoder = LabelEncoder()
df['Department_Label'] = label_encoder.fit_transform(df['Department'])

print("Label Encoded Data:")
print(df)
print()

# Boxplot for Salary
plt.figure()
plt.boxplot(df['Salary'])
plt.ylabel("Salary")
plt.title("Salary Boxplot")
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# Comment on whether Zara’s salary should be treated as an outlier or not.
print("Comment on Zara's Salary:")
print("Zara's salary of 2,000,000 is far above the rest of the team (50,000-90,000), making it a clear outlier.")