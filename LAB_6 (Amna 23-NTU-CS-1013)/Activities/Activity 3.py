# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 6
# Activity 3: Random Forest Regression with GridSearchCV

# Predict house prices using a Random Forest Regressor:
# clean and encode the data, train a basic forest, tune it with GridSearchCV,
# evaluate the best model, compare the default and optimized models,
# and plot actual vs predicted prices and the feature importances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#1. ---Load the dataset---
data = pd.read_csv("train.csv")

#2. ---Display the shape, first 10 rows, missing values and summary statistics of SalePrice---
print("Dataset shape:", data.shape)

print("\nFirst 10 rows:")
print(data.head(10))

missing = data.isnull().sum()
print("\nMissing values (columns with missing values only):")
print(missing[missing > 0].sort_values(ascending=False))

print("\nSummary statistics of SalePrice:")
print(data['SalePrice'].describe())

#3. ---Separate the input features from the target variable and remove the Id column---
X = data.drop(['SalePrice', 'Id'], axis=1)
y = data['SalePrice']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

#4. ---Handle missing values and convert categorical variables into numerical features---
#4.1. Handle missing values
for column in X.columns:
    if pd.api.types.is_numeric_dtype(X[column]):
        X[column] = X[column].fillna(X[column].median())
    else:
        X[column] = X[column].fillna('None')

print("Missing values after cleaning:", X.isnull().sum().sum())

#4.2. Convert categorical variables into numerical features
categorical_features = X.select_dtypes(exclude=['number']).columns
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

print("Features shape after encoding:", X.shape)

#5. ---Split the dataset into training and testing sets (80/20)---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

#6. ---Train a basic Random Forest Regressor (default parameters)---
rf_basic = RandomForestRegressor(random_state=42)
rf_basic.fit(X_train, y_train)
y_pred_basic = rf_basic.predict(X_test)

basic_mae = mean_absolute_error(y_test, y_pred_basic)
basic_mse = mean_squared_error(y_test, y_pred_basic)
basic_rmse = np.sqrt(basic_mse)
basic_r2 = r2_score(y_test, y_pred_basic)

print("\nBasic Random Forest Performance")
print("MAE:", round(basic_mae, 2))
print("MSE:", round(basic_mse, 2))
print("RMSE:", round(basic_rmse, 2))
print("R2 Score:", round(basic_r2, 4))

#7. ---Define the parameter grid---
grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [0.7, 1.0]
}

#8. ---Apply GridSearchCV with 5-fold cross-validation---
rf = RandomForestRegressor(random_state=42)

rf_cv = GridSearchCV(rf, grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_cv.fit(X_train, y_train)

#9. ---Display the best hyperparameters and the best cross-validation score---
print("\nBest Hyperparameters:", rf_cv.best_params_)
print("Best Cross-Validation Score (negative MSE):", round(rf_cv.best_score_, 2))

# Convert the negative MSE into a positive MSE and take its square root to get RMSE
cv_mse = -rf_cv.best_score_
cv_rmse = np.sqrt(cv_mse)
print("Cross-Validation MSE:", round(cv_mse, 2))
print("Cross-Validation RMSE:", round(cv_rmse, 2))

#10. ---Evaluate the best model on the test set---
best_rf = rf_cv.best_estimator_
y_pred_best = best_rf.predict(X_test)

best_mae = mean_absolute_error(y_test, y_pred_best)
best_mse = mean_squared_error(y_test, y_pred_best)
best_rmse = np.sqrt(best_mse)
best_r2 = r2_score(y_test, y_pred_best)

print("\nOptimized Random Forest Performance")
print("MAE:", round(best_mae, 2))
print("MSE:", round(best_mse, 2))
print("RMSE:", round(best_rmse, 2))
print("R2 Score:", round(best_r2, 4))

#11. ---Compare the default and optimized models in a table---
comparison = pd.DataFrame({
    'Model': ['Default Random Forest', 'Optimized Random Forest'],
    'MAE': [round(basic_mae, 2), round(best_mae, 2)],
    'MSE': [round(basic_mse, 2), round(best_mse, 2)],
    'RMSE': [round(basic_rmse, 2), round(best_rmse, 2)],
    'R2 Score': [round(basic_r2, 4), round(best_r2, 4)]
})

print("\nComparison of the default and optimized models:")
print(comparison.to_string(index=False))

#12. ---Plot actual vs predicted prices and the five most influential features---
#12.1. Actual vs predicted house prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()

#12.2. Feature importances of the five most influential features
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)

print("\nFive most influential features:")
print(top_features)

plt.figure(figsize=(8, 5))
top_features.sort_values().plot(kind='barh')
plt.title('Top 5 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()