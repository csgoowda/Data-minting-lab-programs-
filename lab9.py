# Question 9: Apply pruning (Reduced Error Pruning idea) and report pruned model accuracy.
# Aim: Simplify Decision Tree and compare performance before vs after pruning.

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

# Original Decision Tree
model = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=5)

print("Original Accuracy Scores:", scores)
print("Average Accuracy:", scores.mean())

# Pruned Decision Tree (cost complexity pruning parameter)
pruned_model = DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
pruned_scores = cross_val_score(pruned_model, X, y, cv=5)

print("Pruned Accuracy Scores:", pruned_scores)
print("Average Accuracy (Pruned):", pruned_scores.mean())
