# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 3
# Activity 1: Customer Purchase Prediction

# Predict whether a customer will purchase a product using Logistic Regression:
# Preprocess categorical variables, train the model, evaluate using accuracy, precision,
# recall, F1-score, and plot the confusion matrix and ROC curve

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Note: Dataset changed to Social_Network_Ads.csv
# Purchased (0/1) plays the role of Churn.

#1. ---Load and explore dataset---
data = pd.read_csv("Social_Network_Ads.csv")
print(data.head())
print(data.columns)
print()

# Encode categorical features
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Features and target
X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#2. ---Train Logistic Regression model---
model = LogisticRegression()
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

#3. ---Evaluate using Accuracy, Precision, Recall, F1-score---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

#4. ---Plot Confusion Matrix & ROC Curve---
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color="red", label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="black", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()