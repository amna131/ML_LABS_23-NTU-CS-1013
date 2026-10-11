# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab Task no: 1

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Load the Dataset
wine = load_wine()
X, y = wine.data, wine.target

#Split the Dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Define k values to test
k_values = [1,3,5,7,9]

#Train a KNN Classifier for each value of k
#Record the Accuracy Scores
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    accuracy_scores.append(accuracy)

#Plot accuracy against k values
plt.scatter(k_values, accuracy_scores)
plt.xlabel("k values")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k values")
plt.show()

k_values = range(1,30)
accuracy_scores = []

# Apply KNN for different values of k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Print the best value of k
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print("Best value of k:", best_k)
