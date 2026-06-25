import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("dataset.csv")

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Features and Target
X = data.drop("credit_risk", axis=1)
y = data["credit_risk"]

# Original Decision Tree
model = DecisionTreeClassifier()

scores = cross_val_score(model, X, y, cv=5)

print("Original Accuracy Scores:", scores)
print("Average Accuracy:", scores.mean())

# Pruned Decision Tree
pruned_model = DecisionTreeClassifier(ccp_alpha=0.01)

pruned_scores = cross_val_score(pruned_model, X, y, cv=5)

print("Pruned Accuracy Scores:", pruned_scores)
print("Average Accuracy (Pruned):", pruned_scores.mean())
