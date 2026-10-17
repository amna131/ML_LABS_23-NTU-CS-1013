# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab Task 2: Voting Classifier

# Combine Logistic Regression, Random Forest and SVC using a hard voting classifier
# on the Digits dataset and compare the accuracy of each model with the voting result

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#1. Load the Digits dataset
digits = load_digits()
X, y = digits.data, digits.target

#2. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3. Create the classifiers and the voting classifier
log_clf = LogisticRegression(max_iter=1000)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard'
)

#4. Train and compare accuracy of each classifier
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

#5. Predict one digit using the voting classifier
some_digit = X_test[0]
print("Predicted digit:", voting_clf.predict([some_digit]))
print("Actual digit:", y_test[0])