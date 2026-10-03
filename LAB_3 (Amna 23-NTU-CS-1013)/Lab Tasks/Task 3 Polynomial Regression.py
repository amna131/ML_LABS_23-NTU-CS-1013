# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab Task 3: Polynomial Regression

# Fit a degree-4 polynomial regression model on Position_Salaries data
# Plot the fitted curve
# Evaluate using R2 score and RMSE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

#1. Load dataset
data = pd.read_csv("Position_Salaries.csv")

X = data[["Level"]].values
y = data["Salary"].values

#2. Polynomial transformation
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

#3. Train the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

#4. Predictions
y_pred = poly_model.predict(X_poly)

#5. Plot the results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Polynomial Fit (deg=4)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

#6. Evaluation
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("R2 Score:", r2)
print("RMSE:", rmse)
