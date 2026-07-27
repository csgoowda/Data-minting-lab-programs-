# Question 4: Classify all examples and report % correctly classified; explain why not always 100%.
# Aim: Evaluate Decision Tree performance and basic error measures.

# WEKA Steps (Simple)

# 1. Open WEKA Explorer.
# 2. Click Open File.
# 3. Load credit-g.arff (or your GermanCredit dataset).
# 4. In Preprocess, make sure the Class is credit_risk (or class).
# 5. Go to the Classify tab.
# 6. Click Choose → trees → J48.
# 7. Select Percentage Split = 80%.
# 8. Click Start.
# 9. WEKA displays:
#    - Correctly Classified Instances
#    - Incorrectly Classified Instances
#    - Accuracy
#    - MAE
#    - RMSE
#    - Confusion Matrix
# 10. To see the tree:
#     - Right-click the result.
#     - Click Visualize Tree.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

# Load Dataset
data = pd.read_csv("GermanCredit.csv")   # Change path if needed

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

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Correctly Classified:", accuracy * 100, "%")
print("Incorrectly Classified:", (1 - accuracy) * 100, "%")

# Error Measures
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred) ** 0.5)

# Why Can't We Get 100% Accuracy?
# - The dataset may contain noise.
# - Some records have overlapping patterns.
# - Some attributes may not fully explain the target.
# - The model may not perfectly generalize to every example.

# Viva Questions & Answers

# 1. What is Accuracy?
# Answer: Accuracy is the percentage of correctly classified instances.

# 2. How is Accuracy calculated?
# Answer: Accuracy = (Correct Predictions / Total Predictions) × 100.

# 3. What is a correctly classified instance?
# Answer: A record whose predicted class matches the actual class.

# 4. What is an incorrectly classified instance?
# Answer: A record whose predicted class does not match the actual class.

# 5. What is MAE?
# Answer: Mean Absolute Error. It measures the average prediction error.

# 6. What is RMSE?
# Answer: Root Mean Squared Error. It measures the prediction error, giving more weight to larger errors.

# 7. Which is better: High or Low MAE?
# Answer: Lower MAE is better.

# 8. Which is better: High or Low RMSE?
# Answer: Lower RMSE is better.

# 9. Why can't the model achieve 100% accuracy?
# Answer: Because of noise, overlapping data, and the complexity of real-world datasets.

# 10. What is accuracy_score()?
# Answer: It calculates the classification accuracy.

# 11. What is y_test?
# Answer: The actual class labels of the test data.

# 12. What is y_pred?
# Answer: The class labels predicted by the model.

# 13. Why do we compare y_test and y_pred?
# Answer: To evaluate the performance of the model.

# 14. How can we improve accuracy?
# Answer: By using better features, tuning model parameters, cleaning the data,
# or applying cross-validation and pruning.

# 15. What is the conclusion of this experiment?
# Answer: The Decision Tree classified most instances correctly, but 100%
# accuracy is usually not possible because of the nature of the dataset
# and model limitations.
