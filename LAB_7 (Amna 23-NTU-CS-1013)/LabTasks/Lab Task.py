# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 7
# Lab Task: Ensemble Models on the Electricity Dataset

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

elec_data = fetch_openml(name='electricity', version=1)
elec_df = elec_data.frame

print("Data shape:", elec_data.data.shape)
print("Feature names:", elec_data.feature_names)
print("Target names:", elec_data.target_names)

train_data, test_data = train_test_split(
    elec_df, test_size=0.25, random_state=20
)

print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))

print(train_data.head(10))
train_data.info()
print(train_data.describe())
print(train_data.isnull().sum())

corr = train_data.corr(numeric_only=True)

plt.figure(figsize=(12, 7))
sns.heatmap(corr, annot=True, cmap='crest')
plt.show()

plt.figure(figsize=(12, 7))
sns.countplot(data=train_data, x='class')
plt.show()

plt.figure(figsize=(13, 8))
sns.countplot(data=train_data, x='day', hue='class')
plt.show()

train_data.hist(bins=50, figsize=(15, 10))
plt.show()

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

label_enc = LabelEncoder()
y_train_prepared = label_enc.fit_transform(y_train)

print(label_enc.classes_)

def accuracy(model, data, labels):
    predictions = model.predict(data)
    acc = accuracy_score(labels, predictions)
    return acc

X_train_num = X_train.copy()
X_train_num['day'] = X_train_num['day'].astype(int)

models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
    ('svc', SVC(probability=True, random_state=42, kernel='linear'))
]

voting_hard = VotingClassifier(models, voting='hard')
voting_soft = VotingClassifier(models, voting='soft')

print("Starting Voting Classifier", flush=True)

print("Training Hard Voting...", flush=True)
voting_hard.fit(X_train_num, y_train_prepared)
print("Hard Voting completed", flush=True)
print("Hard Voting accuracy:", accuracy(voting_hard, X_train_num, y_train_prepared))

print("Training Soft Voting...", flush=True)
voting_soft.fit(X_train_num, y_train_prepared)
print("Soft Voting completed", flush=True)
print("Soft Voting accuracy:", accuracy(voting_soft, X_train_num, y_train_prepared))

print("Starting Bagging...", flush=True)

bag_classifier = BaggingClassifier(
    DecisionTreeClassifier(class_weight='balanced'),
    max_samples=0.5,
    max_features=0.5,
    bootstrap=False,
    n_estimators=30
)

bag_classifier.fit(X_train_num, y_train_prepared)
print("Bagging accuracy:", accuracy(bag_classifier, X_train_num, y_train_prepared))

print("Starting Gradient Boosting...", flush=True)

grad_boost_clf = GradientBoostingClassifier(
    n_estimators=30,
    learning_rate=0.8,
    random_state=42,
    max_depth=2
)

grad_boost_clf.fit(X_train_num, y_train_prepared)
print("Gradient Boosting accuracy:", accuracy(grad_boost_clf, X_train_num, y_train_prepared))

print("Starting AdaBoost...", flush=True)

adaboost_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=3,
        class_weight='balanced'
    ),
    n_estimators=30,
    learning_rate=0.5
)

adaboost_clf.fit(X_train_num, y_train_prepared)
print("AdaBoost accuracy:", accuracy(adaboost_clf, X_train_num, y_train_prepared))

print("Starting XGBoost...", flush=True)

xgb_clf = xgb.XGBClassifier(n_estimators=30, n_jobs=-1)
xgb_clf.fit(X_train_num, y_train_prepared)
print("XGBoost accuracy:", accuracy(xgb_clf, X_train_num, y_train_prepared))

print("Starting Stacking...", flush=True)

base_models = [
    ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000)),
]

stack_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

stack_clf.fit(X_train_num, y_train_prepared)
print("Stacking accuracy:", accuracy(stack_clf, X_train_num, y_train_prepared))

print("Preparing test data...", flush=True)

X_test = test_data.drop('class', axis=1)
y_test = test_data['class']
y_test_prepared = label_enc.transform(y_test)

X_test_num = X_test.copy()
X_test_num['day'] = X_test_num['day'].astype(int)

print("Test results:", flush=True)
print("Gradient Boosting:", accuracy(grad_boost_clf, X_test_num, y_test_prepared))
print("Bagging:", accuracy(bag_classifier, X_test_num, y_test_prepared))
print("Hard Voting:", accuracy(voting_hard, X_test_num, y_test_prepared))
print("Soft Voting:", accuracy(voting_soft, X_test_num, y_test_prepared))
print("XGBoost:", accuracy(xgb_clf, X_test_num, y_test_prepared))
print("Stacking:", accuracy(stack_clf, X_test_num, y_test_prepared))
print("AdaBoost:", accuracy(adaboost_clf, X_test_num, y_test_prepared))