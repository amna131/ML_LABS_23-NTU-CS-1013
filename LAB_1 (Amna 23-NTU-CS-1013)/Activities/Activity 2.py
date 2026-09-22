# Name: Amna
# Reg no: 23-NTU-CS-1013
# Activity 2: Student Performance Tracker
# Track student performance: handle missing marks, encode grades, visualize scores, and detect outliers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Generate random data for 100 students
np.random.seed(42)

data = {
    "ID": range(1, 101),
    "Math": np.random.randint(40, 101, 100),
    "Science": np.random.randint(40, 101, 100),
    "English": np.random.randint(40, 101, 100),
    "Grade": np.random.choice(["A", "B", "C"], 100)
}

df = pd.DataFrame(data)

# Add some missing marks
df.loc[5, "Math"] = None
df.loc[20, "Science"] = None
df.loc[50, "English"] = None

print("Original Data:")
print(df)
print()

# Handle missing marks using mean
df["Math"] = df["Math"].fillna(df["Math"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())
df["English"] = df["English"].fillna(df["English"].mean())

print("Data after handling missing values:")
print(df)
print()

# Encode Grade column
label_encoder = LabelEncoder()
df["Grade_Label"] = label_encoder.fit_transform(df["Grade"])

print("Encoded Grade Data:")
print(df)
print()

# Histogram of Math scores
plt.hist(df["Math"], bins=10)
plt.xlabel("Math Scores")
plt.ylabel("Number of Students")
plt.title("Math Scores Histogram")
plt.show()

# Calculate total score
df["Total"] = df["Math"] + df["Science"] + df["English"]

# Boxplot of Total Scores
plt.boxplot(df["Total"])
plt.ylabel("Total Score")
plt.title("Total Score Boxplot")
plt.show()

# Detect unusually high or low total scores using IQR method
Q1 = df["Total"].quantile(0.25)
Q3 = df["Total"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["Total"] < lower) | (df["Total"] > upper)]
print("Unusually High/Low Total Scores:")
if len(outliers) == 0:
    print("No student has an unusually high or low total score.")
else:
    print(outliers[["ID", "Total"]])
print()

# Summarize findings: average total score by grade
grade_average = df.groupby("Grade")["Total"].mean()

print("Average Total Score by Grade:")
print(grade_average)
print()
print("Grade with highest average score:", grade_average.idxmax())