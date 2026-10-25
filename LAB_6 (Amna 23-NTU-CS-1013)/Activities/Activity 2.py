# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 6
# Activity 2: Random Forest Classification with GridSearchCV

# Predict whether a patient has heart disease using a Random Forest Classifier:
# clean and encode the data, train a basic forest, tune it with GridSearchCV,
# evaluate the best model on the test set and plot the feature importances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

#1. ---Load the dataset and display its shape, first 10 rows, missing values and class distribution---
data = pd.read_csv("heart_disease_uci.csv")

print("Dataset shape:", data.shape)
print("\nFirst 10 rows:")
print(data.head(10))

print("\nMissing values:")
print(data.isnull().sum())

# 'num' is the target: 0 = no heart disease, 1-4 = heart disease
data['target'] = (data['num'] > 0).astype(int)
data = data.drop(columns=['num', 'id'])

print("\nTarget class distribution:")
print(data['target'].value_counts())

#2. ---Clean the dataset: handle missing values and encode categorical features---
#2.1. Handle missing values
for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):
        data[column] = data[column].fillna(data[column].median())
    else:
        data[column] = data[column].fillna(data[column].mode()[0])

print("\nMissing values after cleaning:", data.isnull().sum().sum())

#2.2. Encode categorical features
categorical_features = data.select_dtypes(exclude=['number']).columns
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

print("Dataset shape after encoding:", data.shape)

#3. ---Separate the input features X from the target variable y---
X = data.drop('target', axis=1)
y = data['target']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

#4. ---Split the data into training and testing sets (80/20)---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#5. ---Train a basic Random Forest Classifier (default parameters)---
rf_basic = RandomForestClassifier(random_state=42)
rf_basic.fit(X_train, y_train)

basic_accuracy = rf_basic.score(X_test, y_test)
print("\nBasic Random Forest Test Accuracy:", round(basic_accuracy, 4))

#6. ---Define the parameter grid for GridSearchCV---
grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

#7. ---Apply GridSearchCV with 5-fold cross-validation---
rf = RandomForestClassifier(random_state=42)

rf_cv = GridSearchCV(rf, grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_cv.fit(X_train, y_train)

#8. ---Display the best hyperparameters and the best cross-validation accuracy---
print("\nBest Hyperparameters:", rf_cv.best_params_)
print("Best Cross-Validation Accuracy:", round(rf_cv.best_score_, 4))

#9. ---Evaluate the best model on the test set---
best_rf = rf_cv.best_estimator_
y_pred = best_rf.predict(X_test)

print("\nBest Model Test Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))

plt.figure(figsize=(8, 6))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#10. ---Plot the feature importances and identify the three most influential features---
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nThree most influential features:")
print(importances.head(3))

plt.figure(figsize=(10, 6))
importances.head(15).sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()