# Question 8: Is it better to prefer simple decision trees over complex trees?
# Aim: Demonstrate impact of tree complexity on underfitting/overfitting and bias-variance.

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

# Compare simple vs complex trees
simple_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
complex_tree = DecisionTreeClassifier(max_depth=None, random_state=42)

simple_scores = cross_val_score(simple_tree, X, y, cv=5)
complex_scores = cross_val_score(complex_tree, X, y, cv=5)

print("Simple Tree (max_depth=3) Mean CV Accuracy:", simple_scores.mean())
print("Complex Tree (max_depth=None) Mean CV Accuracy:", complex_scores.mean())

print("\nObservation:")
print("- Simpler trees usually have higher bias and lower variance.")
print("- Complex trees usually have lower bias and higher variance.")
