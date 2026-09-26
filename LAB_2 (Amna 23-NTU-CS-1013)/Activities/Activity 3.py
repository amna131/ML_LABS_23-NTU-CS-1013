# Name: Amna
# Reg no: 23-NTU-CS-1013
# Activity 3: Used Car Price Prediction

# Predict used car prices using horsepower, engine size, mileage, and curb weight:
# apply Linear Regression and Polynomial Regression with degree 2, 3, and 4,
# compare their performance, and plot fitted curves for different polynomial degrees

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

#1. ---Load and explore dataset---
data = pd.read_csv("Car Price Prediction.csv")
print(data.head())
print(data.columns)
print()

# Note: dataset does not contain Kms_Driven/Year/Engine as named in the activity;
# using enginesize, citympg, curbweight, and horsepower as the closest available features
# Features and target
X = data[["horsepower", "enginesize", "citympg", "curbweight"]].values
y = data["price"].values

#2. ---Apply Linear Regression---
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

r2_linear = r2_score(y, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))

print("Linear Regression")
print("R2 Score:", r2_linear)
print("RMSE:", rmse_linear)
print()

#3. ---Apply Polynomial Regression---

# Degree 2
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)
poly_model2 = LinearRegression()
poly_model2.fit(X_poly2, y)
y_pred2 = poly_model2.predict(X_poly2)
r2_2 = r2_score(y, y_pred2)
rmse_2 = np.sqrt(mean_squared_error(y, y_pred2))
print("Polynomial Regression Degree 2")
print("R2 Score:", r2_2)
print("RMSE:", rmse_2)
print()

# Degree 3
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)
poly_model3 = LinearRegression()
poly_model3.fit(X_poly3, y)
y_pred3 = poly_model3.predict(X_poly3)
r2_3 = r2_score(y, y_pred3)
rmse_3 = np.sqrt(mean_squared_error(y, y_pred3))
print("Polynomial Regression Degree 3")
print("R2 Score:", r2_3)
print("RMSE:", rmse_3)
print()

# Degree 4
poly4 = PolynomialFeatures(degree=4)
X_poly4 = poly4.fit_transform(X)
poly_model4 = LinearRegression()
poly_model4.fit(X_poly4, y)
y_pred4 = poly_model4.predict(X_poly4)
r2_4 = r2_score(y, y_pred4)
rmse_4 = np.sqrt(mean_squared_error(y, y_pred4))
print("Polynomial Regression Degree 4")
print("R2 Score:", r2_4)
print("RMSE:", rmse_4)
print()

#3.1 ---Compare all models together---
print("Model Comparison")
print(f"Linear Regression       R2: {r2_linear:.4f}, RMSE: {rmse_linear:.2f}")
print(f"Polynomial Degree 2     R2: {r2_2:.4f}, RMSE: {rmse_2:.2f}")
print(f"Polynomial Degree 3     R2: {r2_3:.4f}, RMSE: {rmse_3:.2f}")
print(f"Polynomial Degree 4     R2: {r2_4:.4f}, RMSE: {rmse_4:.2f}")

#4. ---Plot fitted curves for different polynomial degrees---
# Plotted against horsepower (other features held at their mean value)
hp_col = X[:, 0]
mean_eng = X[:, 1].mean()
mean_mpg = X[:, 2].mean()
mean_wt = X[:, 3].mean()

hp_range = np.linspace(hp_col.min(), hp_col.max(), 100).reshape(-1, 1)
X_curve = np.hstack([
    hp_range,
    np.full_like(hp_range, mean_eng),
    np.full_like(hp_range, mean_mpg),
    np.full_like(hp_range, mean_wt)
])

y_linear_curve = linear_model.predict(X_curve)
y_poly2_curve = poly_model2.predict(poly2.transform(X_curve))
y_poly3_curve = poly_model3.predict(poly3.transform(X_curve))
y_poly4_curve = poly_model4.predict(poly4.transform(X_curve))

plt.scatter(hp_col, y, label="Actual Data")
plt.plot(hp_range, y_linear_curve, label="Linear Regression")
plt.plot(hp_range, y_poly2_curve, label="Polynomial Degree 2")
plt.plot(hp_range, y_poly3_curve, label="Polynomial Degree 3")
plt.plot(hp_range, y_poly4_curve, label="Polynomial Degree 4")

plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Used Car Price Prediction")
plt.legend()
plt.show()