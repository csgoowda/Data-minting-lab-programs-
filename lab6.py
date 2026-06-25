# Question 6: Train Decision Tree using selected attributes and compare performance.
# Aim: Check if fewer selected attributes can still provide good results.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("credit-g_csv.csv")

# Select required attributes by index if present in dataset shape
# Original suggestion: 2,3,5,7,10,17 and class attribute
selected_indices = [1, 2, 4, 6, 9, 15]

# Identify target column
target_col = 'class' if 'class' in data.columns else 'credit_risk'

# Build selected dataframe safely
feature_cols = [data.columns[i] for i in selected_indices if i < len(data.columns)]
feature_cols = [c for c in feature_cols if c != target_col]

data = data[feature_cols + [target_col]]

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Features and Target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Calculate Accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)
print("Correctly Classified:", accuracy * 100, "%")
print("Wrongly Classified:", (1 - accuracy) * 100, "%")
