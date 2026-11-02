# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 7
# Activity 3: Stacking for Regression
# Test whether stacking beats each single regression model
# Predict medical insurance charges with Ridge, Random Forest and Gradient Boosting,
# stack them in a StackingRegressor, and compare all four models (MAE, RMSE, R2)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the dataset
insurance = pd.read_csv("insurance.csv")

print("Dataset shape:", insurance.shape)

print("\nFirst 10 rows:")
print(insurance.head(10))

print("\nMissing values:")
print(insurance.isnull().sum())

print("\nSummary statistics of charges:")
print(insurance["charges"].describe())


# 2. Encode categorical columns
insurance["sex"] = insurance["sex"].map({"female": 0, "male": 1})
insurance["smoker"] = insurance["smoker"].map({"no": 0, "yes": 1})

insurance = pd.get_dummies(insurance, columns=["region"], drop_first=True)

X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Train the three base models separately

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_model = Ridge()
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

ridge_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
gb_model.fit(X_train_scaled, y_train)


# 4. Build and train the Stacking Regressor
estimators = [
    ("ridge", ridge_model),
    ("random_forest", rf_model),
    ("gradient_boosting", gb_model)
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5
)

stack_model.fit(X_train_scaled, y_train)


# 5. Compare all four models
models = {
    "Ridge": ridge_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Stacking": stack_model
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)

    results.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "R2": round(r2_score(y_test, y_pred), 4)
    })

results_df = pd.DataFrame(results)

print("\nComparison of all models:")
print(results_df.to_string(index=False))


# 6. Print coefficients of the final estimator
base_names = ["Ridge", "Random Forest", "Gradient Boosting"]
coefficients = stack_model.final_estimator_.coef_

print("\nFinal estimator coefficients:")

for name, coefficient in zip(base_names, coefficients):
    print(name, ":", round(coefficient, 4))

print("Intercept:", round(stack_model.final_estimator_.intercept_, 2))

# The base model with the largest coefficient is the one the stack trusts most
print("The stack trusts", base_names[int(np.argmax(coefficients))], "the most")


# Plot actual versus predicted charges for the stacking model
stack_predictions = stack_model.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, stack_predictions, alpha=0.7)

# Reference line for perfect predictions
min_value = min(y_test.min(), stack_predictions.min())
max_value = max(y_test.max(), stack_predictions.max())
plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")

plt.title("Actual vs Predicted Insurance Charges - Stacking")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.tight_layout()
plt.show()