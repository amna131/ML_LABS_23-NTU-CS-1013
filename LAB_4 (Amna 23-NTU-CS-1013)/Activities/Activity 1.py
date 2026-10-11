# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 4
# Activity 1: KNN for Credit Card Fraud Detection

# Detect fraudulent transactions in a highly imbalanced dataset using KNN:
# handle imbalance with SMOTE, scale features, train KNN with k=5,
# tune k using GridSearchCV, and compare Accuracy and F1-score

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

#1. ---Load dataset and split into features X and target y---
data = pd.read_csv("creditcard.csv")

# Using 20% of the data to reduce running time
data = data.sample(frac=0.2, random_state=42)

print(data.head())
print(data['Class'].value_counts())
print()

X = data.drop('Class', axis=1)
y = data['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#2. ---Handle class imbalance using SMOTE (training data only)---
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("Class counts after SMOTE:")
print(y_train_sm.value_counts())
print()

#3. ---Scale features---
scaler = StandardScaler()
X_train_sm = scaler.fit_transform(X_train_sm)
X_test = scaler.transform(X_test)

#4. ---Train KNN classifier with k=5---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sm, y_train_sm)
y_pred = knn.predict(X_test)

acc_5 = accuracy_score(y_test, y_pred)
f1_5 = f1_score(y_test, y_pred)
print("KNN (k=5)")
print("Accuracy:", acc_5)
print("F1-score:", f1_5)
print()

#5. ---Tune k using GridSearchCV---
grid = {
    'n_neighbors': [3, 5, 7, 9]
}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=3, n_jobs=-1)

knn_cv.fit(X_train_sm, y_train_sm)

print("Best Parameters:", knn_cv.best_params_)
print("Train Score:", knn_cv.best_score_)
print("Test Score:", knn_cv.score(X_test, y_test))

y_pred_best = knn_cv.predict(X_test)

acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
print("KNN (tuned k)")
print("Accuracy:", acc_best)
print("F1-score:", f1_best)
print()

#6. ---Compare model performance---
print("Model Comparison")
print(f"KNN (k=5)          Accuracy: {acc_5:.4f}, F1-score: {f1_5:.4f}")
print(f"KNN (k={knn_cv.best_params_['n_neighbors']}, tuned)   Accuracy: {acc_best:.4f}, F1-score: {f1_best:.4f}")