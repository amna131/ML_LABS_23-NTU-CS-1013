# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab Task 1: Support Vector Machine (SVM)

# Predict whether a person's income is above 50K using an SVM classifier:
# encode categorical features, scale numerical features, train SVC,
# and evaluate using cross-validation (train score) and test score

from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings("ignore")


#1. Load and explore dataset
df = pd.read_csv('income_evaluation.csv')
print(df.head())

# Separate categorical (text) and numerical features
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(exclude=['object']).columns

#2.Preprocess
#2.1. Encode categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Features and target
X = df.drop("income_ >50K", axis=1)
y = df["income_ >50K"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#2.2. Scale numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
print(X_train.head())

#3. Train SVM model
svc = SVC(random_state=101)
svc.fit(X_train,y_train)

#4. Evaluate the model
# Train score: average accuracy from 5-fold cross-validation
accuracies = cross_val_score(svc,X_train,y_train,cv=5)
print("Train Score:", np.mean(accuracies))

#5. Test score: accuracy on the unseen test data
print("Test Score:", svc.score(X_test,y_test))