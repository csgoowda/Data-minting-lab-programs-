import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv(r"C:\Users\cheth\Downloads\GermanCredit.csv")

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Features and Target
X = data.drop("credit_risk", axis=1)
y = data["credit_risk"]

# Create Model
model = DecisionTreeClassifier()

# 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("Accuracy Scores:", scores)
print("Average Accuracy:", scores.mean())
