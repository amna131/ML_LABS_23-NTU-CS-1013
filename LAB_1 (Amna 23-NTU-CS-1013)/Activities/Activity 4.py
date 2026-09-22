# Name: Amna
# Reg no: 23-NTU-CS-1013
# Activity 4: Multi-format Data Loading & Visualization Challenge
# Load, merge, and visualize student data from CSV, JSON, and Excel files plus one-hot encode Name

import pandas as pd
import matplotlib.pyplot as plt

# Load students data from CSV
students_df = pd.read_csv('students.csv')
print("Students Data (CSV):")
print(students_df.head())
print()

# Load attendance data from JSON
attendance_df = pd.read_json('attendance.json')
print("Attendance Data (JSON):")
print(attendance_df.head())
print()

# Load bonus marks from Excel
extra_df = pd.read_excel('extra.xlsx')
print("Extra/Bonus Marks Data (Excel):")
print(extra_df.head())
print()

# Merge all three DataFrames into one, on the "Name" column
merged_df = students_df.merge(attendance_df, on="Name").merge(extra_df, on="Name")
print("Merged Data:")
print(merged_df)
print()

# Scatter plot: Marks vs Attendance (highlight attendance below 70%)
low = merged_df[merged_df["Attendance"] < 70]
normal = merged_df[merged_df["Attendance"] >= 70]

plt.scatter(normal["Attendance"], normal["Marks"], color="blue", label="Attendance >= 70%")
plt.scatter(low["Attendance"], low["Marks"], color="red", label="Attendance < 70%")
plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.title("Marks vs Attendance")
plt.legend()
plt.show()

# One-hot encode the Name column
merged_encoded = pd.get_dummies(merged_df, columns=["Name"])
print("One-Hot Encoded Data:")
print(merged_encoded)