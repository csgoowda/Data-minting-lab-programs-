# Question 8: Is it better to prefer simple decision trees over complex trees?
# Aim: Demonstrate impact of tree complexity on underfitting/overfitting and bias-variance.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("GermanCredit.csv")

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Features and Target
X = data.drop("credit_risk", axis=1)
y = data["credit_risk"]

# Simple Decision Tree
simple_tree = DecisionTreeClassifier(max_depth=3, random_state=42)

# Complex Decision Tree
complex_tree = DecisionTreeClassifier(random_state=42)

# Cross Validation
simple_acc = cross_val_score(simple_tree, X, y, cv=5).mean()
complex_acc = cross_val_score(complex_tree, X, y, cv=5).mean()

print("Simple Tree Accuracy :", simple_acc)
print("Complex Tree Accuracy:", complex_acc)

print("\nConclusion:")
print("Simple Tree -> Higher Bias, Lower Variance")
print("Complex Tree -> Lower Bias, Higher Variance")

# What is the Question Asking?
# The examiner wants to know:
# - Is a simple tree better than a complex tree?
# - How does tree complexity affect bias and variance?

# Answer
# A simple Decision Tree is easy to understand and less likely to overfit.
# It has higher bias and lower variance.
# A complex Decision Tree can fit the training data very well but may overfit.
# It has lower bias and higher variance.
# Therefore, a balanced (or pruned) Decision Tree is usually preferred.

# Viva Questions

# 1. What is a simple Decision Tree?
# Answer: A tree with fewer levels and branches.

# 2. What is a complex Decision Tree?
# Answer: A tree with many levels and branches.

# 3. Which tree is generally preferred?
# Answer: A simple or properly pruned Decision Tree.

# 4. Why?
# Answer: Because it reduces overfitting and is easier to understand.

# 5. What is bias?
# Answer: Error caused by overly simple assumptions in the model.

# 6. What is variance?
# Answer: The model's sensitivity to changes in the training data.

# 7. Which tree has higher bias?
# Answer: A simple Decision Tree.

# 8. Which tree has higher variance?
# Answer: A complex Decision Tree.

# 9. What is overfitting?
# Answer: When a model learns the training data too closely and performs
# poorly on new data.

# 10. What is the conclusion?
# Answer: A balanced or pruned Decision Tree provides the best trade-off
# between bias and variance.

print("\nObservation:")
print("- Simpler trees usually have higher bias and lower variance.")
print("- Complex trees usually have lower bias and higher variance.")
