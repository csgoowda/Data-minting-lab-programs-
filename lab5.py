# Question 5: Explain cross-validation and train a Decision Tree using cross-validation.
# Aim: Evaluate model robustness using 5-fold cross-validation.

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

# Decision Tree
model = DecisionTreeClassifier(random_state=42)

# 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

# Output
print("Accuracy of Each Fold:")
print(scores)

print("\nAverage Accuracy:")
print(scores.mean())


# What is Cross Validation?
# Cross Validation is a model evaluation technique where the dataset is
# divided into k equal parts (folds). The model is trained and tested
# k times, using a different fold for testing each time.
# The final accuracy is the average of all folds.

# Why use Cross Validation?
# - Uses all the data for both training and testing.
# - Gives a more reliable accuracy.
# - Reduces bias from a single train-test split.

# Viva Questions

# 1. What is Cross Validation?
# Answer: Cross Validation is a technique used to evaluate a machine
# learning model by dividing the dataset into multiple folds.

# 2. What is 5-Fold Cross Validation?
# Answer: The dataset is divided into 5 equal parts. Four parts are
# used for training and one part for testing. This is repeated 5 times.

# 3. Why do we use Cross Validation?
# Answer: To obtain a more reliable estimate of the model's performance.

# 4. What does cv=5 mean?
# Answer: It performs 5-Fold Cross Validation.

# 5. Which function performs Cross Validation?
# Answer: cross_val_score().

# 6. What is a fold?
# Answer: A fold is one part of the dataset used in Cross Validation.

# 7. Does Cross Validation always increase accuracy?
# Answer: No. It provides a more reliable evaluation; the accuracy may
# increase, decrease slightly, or remain similar.

# 8. What is the main advantage of Cross Validation?
# Answer: It uses all the data for evaluation and reduces evaluation bias.

# 9. How many times is the model trained in 5-Fold Cross Validation?
# Answer: Five times.

# 10. Which classifier did you use?
# Answer: Decision Tree Classifier.
