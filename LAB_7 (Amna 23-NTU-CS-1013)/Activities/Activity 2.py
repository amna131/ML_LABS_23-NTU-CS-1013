# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 7
# Activity 2: Boosting
# Train and tune boosting models on an imbalanced problem.
# Predict customer churn using Gradient Boosting, AdaBoost and XGBoost,
# tune XGBoost with GridSearchCV, and compare default and tuned models.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from xgboost import XGBClassifier


# 1. Load the dataset
telco = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset shape:", telco.shape)

print("\nFirst 10 rows:")
print(telco.head(10))

print("\nMissing values:")
print(telco.isnull().sum())

print("\nChurn distribution:")
print(telco["Churn"].value_counts())


# 2. Convert TotalCharges to numeric and handle blank values
telco["TotalCharges"] = pd.to_numeric(
    telco["TotalCharges"], errors="coerce"
)

print("\nMissing TotalCharges after conversion:")
print(telco["TotalCharges"].isnull().sum())

telco["TotalCharges"] = telco["TotalCharges"].fillna(
    telco["TotalCharges"].median()
)


# 3. Drop customerID and encode the target
telco = telco.drop("customerID", axis=1)

telco["Churn"] = telco["Churn"].map({"No": 0, "Yes": 1})

# One-hot encode categorical columns
X = telco.drop("Churn", axis=1)
y = telco["Churn"]

X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# 4. Train default boosting models
gb_model = GradientBoostingClassifier(random_state=42)
ada_model = AdaBoostClassifier(random_state=42)
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

gb_model.fit(X_train, y_train)
ada_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

print("\nDefault boosting models trained successfully.")


# 5. Tune XGBoost using GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_

print("\nBest XGBoost parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1-score:")
print(round(grid_search.best_score_, 4))


# 6. Compare default and tuned XGBoost on the test set
models = {
    "Default XGBoost": xgb_model,
    "Tuned XGBoost": best_xgb
}

results = []

for name, model in models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(
            precision_score(y_test, y_pred, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_test, y_pred, zero_division=0), 4
        ),
        "F1-score": round(
            f1_score(y_test, y_pred, zero_division=0), 4
        ),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
    })

results_df = pd.DataFrame(results)

print("\nDefault vs Tuned XGBoost:")
print(results_df.to_string(index=False))


# Plot the five most important features of tuned XGBoost
importance = pd.Series(
    best_xgb.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(5)

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh")
plt.title("Top 5 Feature Importances - Tuned XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Explain why accuracy alone is misleading here.
#The dataset has more customers who stayed than customers who left.
# A model can achieve good accuracy by predicting the majority class
# while missing many customers who actually churned.
# Precision, recall, F1-score, and ROC-AUC provide a more complete evaluation.
