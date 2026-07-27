# Question 12: Implement SVM and compare accuracy with Decision Tree results.
# Aim: Train SVM and Decision Tree on same split and compare performance.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVM Model
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Decision Tree Model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Accuracy
svm_acc = accuracy_score(y_test, svm_pred)
dt_acc = accuracy_score(y_test, dt_pred)

print("SVM Accuracy:", svm_acc)
print("Decision Tree Accuracy:", dt_acc)

# Compare
if svm_acc > dt_acc:
    print("SVM performed better than Decision Tree.")
elif dt_acc > svm_acc:
    print("Decision Tree performed better than SVM.")
else:
    print("Both models have the same accuracy.")


# What is the Question Asking?
# The question asks you to:
# - Train an SVM model.
# - Train a Decision Tree model.
# - Compare their accuracies.
# - Identify which classifier performs better.

# Conclusion
# The SVM and Decision Tree were trained on the same dataset.
# In many cases, SVM achieves slightly higher accuracy because
# it finds an optimal decision boundary, while the Decision Tree
# is easier to interpret. The better model is the one with the
# higher accuracy on your dataset.

# Viva Questions

# 1. What is SVM?
# Answer: SVM (Support Vector Machine) is a supervised machine
# learning algorithm used mainly for classification.

# 2. What is the main objective of SVM?
# Answer: To find the optimal hyperplane that separates different classes.

# 3. What is a hyperplane?
# Answer: A decision boundary that separates different classes.

# 4. What are support vectors?
# Answer: The data points closest to the hyperplane that determine its position.

# 5. Which WEKA classifier represents SVM?
# Answer: SMO (Sequential Minimal Optimization).

# 6. Which WEKA classifier represents the Decision Tree?
# Answer: J48.

# 7. Which classifier performed better?
# Answer: The classifier with the higher accuracy
# (often SVM in this experiment).

# 8. Why compare SVM and Decision Tree?
# Answer: To determine which algorithm gives better
# classification performance on the dataset.

# 9. Which algorithm is easier to interpret?
# Answer: Decision Tree.

# 10. What is the conclusion?
# Answer: Both are supervised classifiers, but SVM often
# provides better accuracy, while Decision Trees are simpler
# to understand and visualize.
