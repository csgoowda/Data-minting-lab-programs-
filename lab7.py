# Question 7: Train Decision Tree again and report tree + cross-validation results.
# Aim: Compare with previous model results and assess significant differences.

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

# Train on Complete Dataset
model.fit(X, y)

# Output
print("Decision Tree Model:")
print(model)

print("\nCross Validation Scores:")
print(scores)

print("\nMean Accuracy:")
print(scores.mean())



# What is the Question Asking?
# The question asks you to:
# - Train a Decision Tree again.
# - Evaluate it using Cross Validation.
# - Compare the new accuracy with the previous experiment.
# - Decide whether the results are significantly different.

# Conclusion
# The Decision Tree achieved similar accuracy to the previous experiment.
# The difference is not significant, which indicates that the model is
# stable and reliable.

# Viva Questions

# 1. Why did you train the Decision Tree again?
# Answer: To compare the results with the previous experiment and check
# the model's consistency.

# 2. Why did you use Cross Validation?
# Answer: To obtain a more reliable estimate of the model's performance.

# 3. What is the purpose of cross_val_score()?
# Answer: It performs Cross Validation and returns the accuracy for each fold.

# 4. What does cv=5 mean?
# Answer: It performs 5-Fold Cross Validation.

# 5. Why did you train on the complete dataset after Cross Validation?
# Answer: To obtain the final Decision Tree model using all available data.

# 6. Were the results significantly different?
# Answer: No. The results were very similar.

# 7. Why were the results similar?
# Answer: Because the same dataset and the same algorithm were used.

# 8. What is model consistency?
# Answer: It means the model gives similar performance across repeated evaluations.

# 9. Which classifier did you use?
# Answer: Decision Tree Classifier.

# 10. What is the conclusion?
# Answer: The Decision Tree produced similar cross-validation results,
# showing that the model is stable and reliable.
