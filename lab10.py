

# Question
# How can you convert a Decision Tree into IF–THEN–ELSE rules?
# Make up your own small Decision Tree consisting of 2–3 levels and
# convert it into a set of rules. There also exist different classifiers
# that output the model in the form of rules. One such classifier in
# WEKA is rules.PART. Train this model and report the set of rules
# obtained. Sometimes just one attribute can be good enough in making
# the decision. Can you predict what attribute that might be in this
# dataset? OneR classifier uses a single attribute to make decisions
# (it chooses the attribute based on minimum error). Report the rule
# obtained by training a OneR classifier. Rank the performance of
# J48, PART, and OneR.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("credit-g_csv.csv")

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Features and Target
X = data.drop("class", axis=1)
y = data["class"]

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Print IF-THEN Rules
rules = export_text(model, feature_names=list(X.columns))
print(rules)









import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("credit-g_csv.csv")

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Features and Target
X = data.drop("class", axis=1)
y = data["class"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# OneR (Depth = 1)
model = DecisionTreeClassifier(max_depth=1)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Selected Attribute:", X.columns[model.tree_.feature[0]])
print("Accuracy:", accuracy_score(y_test, y_pred))


# Question 10 Viva Questions & Answers

# 1. What is a Decision Tree?
# Answer: A Decision Tree is a supervised machine learning algorithm used for classification and prediction.

# 2. What is an IF–THEN rule?
# Answer: An IF–THEN rule is a decision rule where the output depends on one or more conditions.
# Example:
# IF checking_status = good
# THEN credit = good

# 3. How do you convert a Decision Tree into IF–THEN rules?
# Answer: Each path from the root node to a leaf node is converted into one IF–THEN rule.

# 4. What is PART?
# Answer: PART is a rule-based classifier that generates IF–THEN rules from partial Decision Trees.

# 5. Is PART supervised or unsupervised?
# Answer: Supervised.

# 6. Which WEKA classifier generates rules?
# Answer: PART.

# 7. What is OneR?
# Answer: OneR (One Rule) is a simple classifier that makes decisions using only one attribute.

# 8. Why is it called OneR?
# Answer: Because it creates one rule using the best single attribute.

# 9. Which attribute was selected by OneR?
# Answer: checking_status.

# 10. Why was checking_status selected?
# Answer: Because it had the lowest classification error (or highest predictive power)
# among all attributes.

# 11. Which algorithm is the simplest?
# Answer: OneR.

# 12. Which algorithm gives the highest accuracy?
# Answer: J48.

# 13. Which algorithm is easiest to understand?
# Answer: OneR.

# 14. What is the difference between J48 and PART?
# Answer:
# J48 produces a Decision Tree.
# PART produces IF–THEN rules.

# 15. What is the difference between PART and OneR?
# Answer:
# PART uses multiple rules.
# OneR uses only one rule (one attribute).

# 16. Which classifier uses only one attribute?
# Answer: OneR.

# 17. Why are IF–THEN rules useful?
# Answer: They are easy to understand and explain.

# 18. Which WEKA classifier did you use for Part B?
# Answer: rules → PART.

# 19. Which WEKA classifier did you use for Part C?
# Answer: rules → OneR.

# 20. Rank the performance of J48, PART, and OneR.
# Answer:
# J48 – Highest accuracy.
# PART – Slightly lower or similar accuracy.
# OneR – Lowest accuracy but simplest model.
