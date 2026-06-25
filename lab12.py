# Question 12: Implement SVM and compare accuracy with Decision Tree results.
# Aim: Train SVM and Decision Tree on same split and compare performance.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("credit-g_csv.csv")

# Label Encoding
labelencoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = labelencoder.fit_transform(data[column])

# Features and Target
if 'class' in data.columns:
    X = data.drop('class', axis=1)
    y = data['class']
else:
    X = data.drop('credit_risk', axis=1)
    y = data['credit_risk']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVM Classifier
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", accuracy)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)
