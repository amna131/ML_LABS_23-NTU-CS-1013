# Name: Amna
# Reg no: 23-NTU-CS-1013
# Activity 1: Medical Insurance Cost Prediction

# Predict a person's medical insurance cost using Linear Regression:
# preprocess categorical and numerical features, train the model,
# evaluate using RMSE and R2 score, and plot predicted vs actual costs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#1. ---Load and explore dataset---
data = pd.read_csv("Medical Cost Personal Datasets.csv")
print(data.head())
print(data.columns)
print()

#Features and target
X = data[['age', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']

#2. ---Preprocess:---
#2.1.  Encode categorical features
le = LabelEncoder()
X['smoker'] = le.fit_transform(X['smoker'])
X['region'] = le.fit_transform(X['region'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2.2. Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#3. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#4. Evaluate the Model RMSE & R2 Score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)


#5. Plot Predicted vs Actual Costs
plt.scatter(y_test, y_pred, color="green", label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black", linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title("Predicted vs Actual Insurance Costs")
plt.legend()
plt.show()