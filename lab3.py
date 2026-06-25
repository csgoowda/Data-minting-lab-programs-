# Question 3: Train a Decision Tree using the complete dataset as training data and report the model.
# Aim: Train and visualize a Decision Tree classifier.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("credit-g_csv.csv")

# Convert categorical values to numeric
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Model:")
print(model)

print("\nEvaluation Metrics:")
print("Accuracy:", accuracy)
print("Correctly Classified:", accuracy * 100, "%")

# Plot Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['0', '1'],
    filled=True
)
plt.show()
