# Question 9: Apply pruning (Reduced Error Pruning idea) and report pruned model accuracy.
# Aim: Simplify Decision Tree and compare performance before vs after pruning.


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

# Original Decision Tree
tree = DecisionTreeClassifier(random_state=42)
score = cross_val_score(tree, X, y, cv=5)

print("Original Accuracy:", score)
print("Average Accuracy:", score.mean())

# Pruned Decision Tree
pruned_tree = DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
pruned_score = cross_val_score(pruned_tree, X, y, cv=5)

print("\nPruned Accuracy:", pruned_score)
print("Average Accuracy (Pruned):", pruned_score.mean())


# What is the Question Asking?
# The question asks you to:
# - Train a normal Decision Tree.
# - Train a pruned Decision Tree.
# - Compare their accuracies.
# - Explain whether pruning improves the model.

# Theory Answer
# Reduced Error Pruning removes unnecessary branches from a Decision Tree
# if removing them does not reduce the model's accuracy. It simplifies
# the tree, reduces overfitting, and usually improves generalization.

# Conclusion
# The pruned Decision Tree is smaller and simpler than the original tree.
# Its accuracy may increase slightly, remain similar, or decrease slightly,
# but it generally performs better on unseen data because it reduces overfitting.

# Viva Questions

# 1. What is pruning?
# Answer: Pruning is the process of removing unnecessary branches from a Decision Tree.

# 2. Why do we use pruning?
# Answer: To reduce overfitting and simplify the model.

# 3. What is Reduced Error Pruning?
# Answer: It removes branches that do not improve the model's accuracy.

# 4. What is ccp_alpha?
# Answer: It is the pruning parameter in Scikit-learn that controls the amount of pruning.

# 5. What happens if ccp_alpha = 0?
# Answer: No pruning is applied.

# 6. What happens if ccp_alpha is large?
# Answer: The tree becomes smaller and may underfit the data.

# 7. Does pruning always increase accuracy?
# Answer: No. It may increase, remain the same, or decrease slightly,
# but it usually improves generalization.

# 8. What is overfitting?
# Answer: Overfitting occurs when a model learns the training data too
# closely and performs poorly on new data.

# 9. Which classifier did you use?
# Answer: Decision Tree Classifier (J48 in WEKA).

# 10. What is the conclusion of this experiment?
# Answer: Pruning produces a simpler Decision Tree and usually improves
# the model's ability to generalize to new data.
