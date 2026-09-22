# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task 11
# Outlier Visualization using Boxplot
import matplotlib.pyplot as plt
import numpy as np

# Sample data with an outlier
data = [10, 12, 11, 13, 12, 14, 13, 100]  # 100 is an outlier

# Plot boxplot
plt.boxplot(data)
plt.title("Outlier Visualization")
plt.show()