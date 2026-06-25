# Question 7: Train Decision Tree again and report tree + cross-validation results.
# Aim: Compare with previous model results and assess significant differences.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("credit-g_csv.csv")

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Features and Target
if 'class' in data.columns:
    X = data.drop("class", axis=1)
    y = data["class"]
else:
    X = data.drop("credit_risk", axis=1)
    y = data["credit_risk"]

# Create Decision Tree
model = DecisionTreeClassifier()

# Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Train on Full Dataset
model.fit(X, y)

print(model)
print("Cross Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
