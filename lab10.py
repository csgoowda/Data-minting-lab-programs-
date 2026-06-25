# Question 10: Convert Decision Tree to rules, explore rule-based model, and compare with OneR idea.
# Aim: Show IF-THEN rules and compare simple rule-style approaches.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
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

# Train Decision Tree and print rules
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

rules = export_text(dt, feature_names=list(X.columns))
print("Decision Tree IF-THEN style rules:\n")
print(rules)

# Basic performance
dt_scores = cross_val_score(dt, X, y, cv=5)
print("Decision Tree Mean CV Accuracy:", dt_scores.mean())

print("\nNote:")
print("For Weka-specific PART and OneR, run in Weka GUI/CLI and compare with this DT baseline.")
