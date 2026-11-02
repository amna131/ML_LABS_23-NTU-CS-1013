# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 7
# Activity 1: Voting and Bagging
# Compare voting and bagging ensembles with single models.
# Predict water potability using Logistic Regression, SVC and Random Forest,
# combine models with hard and soft voting, and compare bagging and pasting
# using Decision Trees based on accuracy, precision, recall and F1-score.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# 1. Load the CSV and display dataset information
water = pd.read_csv("water_potability.csv")

print("Dataset shape:", water.shape)

print("\nFirst 10 rows:")
print(water.head(10))

print("\nMissing values:")
print(water.isnull().sum())

print("\nClass distribution of Potability:")
print(water["Potability"].value_counts())


# 2. Split the data into training and testing sets
X = water.drop("Potability", axis=1)
y = water["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# 3. Fill missing values using training-set medians
for column in ["ph", "Sulfate", "Trihalomethanes"]:
    median_value = X_train[column].median()
    X_train[column] = X_train[column].fillna(median_value)
    X_test[column] = X_test[column].fillna(median_value)


# 4. Scale data for Logistic Regression and SVC
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train single models
log_clf = LogisticRegression()
svc_clf = SVC(probability=True, random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)

log_clf.fit(X_train_scaled, y_train)
svc_clf.fit(X_train_scaled, y_train)
rnd_clf.fit(X_train, y_train)


# Voting Classifier: Hard and Soft Voting
models = [
    ("lr", LogisticRegression()),
    ("svc", SVC(probability=True, random_state=42)),
    ("rf", RandomForestClassifier(random_state=42))
]

voting_hard = VotingClassifier(models, voting="hard")
voting_soft = VotingClassifier(models, voting="soft")

voting_hard.fit(X_train_scaled, y_train)
voting_soft.fit(X_train_scaled, y_train)


# 5. Bagging and Pasting with Decision Trees
bag_10 = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)

bag_100 = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)

paste_100 = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.5,
    bootstrap=False,
    random_state=42
)

tree_clf = DecisionTreeClassifier(random_state=42)

bag_10.fit(X_train, y_train)
bag_100.fit(X_train, y_train)
paste_100.fit(X_train, y_train)
tree_clf.fit(X_train, y_train)


# 6. Compare all models
all_models = {
    "Logistic Regression": log_clf,
    "SVC": svc_clf,
    "Random Forest": rnd_clf,
    "Decision Tree": tree_clf,
    "Hard Voting": voting_hard,
    "Soft Voting": voting_soft,
    "Bagging (10 trees)": bag_10,
    "Bagging (100 trees)": bag_100,
    "Pasting (100 trees)": paste_100
}

results = []

for name, model in all_models.items():

    if name in ["Logistic Regression", "SVC", "Hard Voting", "Soft Voting"]:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

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
        )
    })

results_df = pd.DataFrame(results)

print("\nComparison of all models:")
print(results_df.to_string(index=False))


# Confusion matrix of the best model (highest accuracy)
best_name = results_df.loc[
    results_df["Accuracy"].idxmax(), "Model"
]

best_model = all_models[best_name]

if best_name in ["Logistic Regression", "SVC", "Hard Voting", "Soft Voting"]:
    best_predictions = best_model.predict(X_test_scaled)
else:
    best_predictions = best_model.predict(X_test)

print("\nBest model:", best_name)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Potable", "Potable"],
    yticklabels=["Not Potable", "Potable"]
)

plt.title("Confusion Matrix of " + best_name)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Compare the best single model with the best ensemble model
single_names = [
    "Logistic Regression",
    "SVC",
    "Random Forest",
    "Decision Tree"
]

ensemble_names = [
    "Hard Voting",
    "Soft Voting",
    "Bagging (10 trees)",
    "Bagging (100 trees)",
    "Pasting (100 trees)"
]

best_single = results_df[
    results_df["Model"].isin(single_names)
].sort_values("Accuracy").iloc[-1]

best_ensemble = results_df[
    results_df["Model"].isin(ensemble_names)
].sort_values("Accuracy").iloc[-1]

print(
    "\nBest single model:",
    best_single["Model"],
    best_single["Accuracy"]
)

print(
    "Best voting/bagging model:",
    best_ensemble["Model"],
    best_ensemble["Accuracy"]
)

# Did voting or bagging beat the best single model? Explain in two sentences?
# Hard Voting achieved an accuracy of 0.6738,
# while the best single model, SVC achieved 0.6707.
# Pasting (100 trees) also achieved a slightly higher accuracy of 0.6723 than SVC.
# Therefore, voting and pasting performed slightly better than the best single model,
# while the two bagging models did not.