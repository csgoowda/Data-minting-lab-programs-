# Question 3: Train a Decision Tree using the complete dataset as training data and report the model.
# Aim: Train and visualize a Decision Tree classifier.

# WEKA Steps
# 1. Open WEKA Explorer.
# 2. Click Open File.
# 3. Select credit-g.arff (or GermanCredit.arff).
# 4. Go to the Classify tab.
# 5. Click Choose.
# 6. Select trees → J48.
# 7. Under Test options, select Percentage Split and enter 80%
#    (or use Cross-validation (10 folds) if your lab asks for it).
# 8. Click Start.
# 9. WEKA will display:
#    - Decision Tree (J48)
#    - Correctly Classified Instances
#    - Incorrectly Classified Instances
#    - Accuracy (%)
# 10. To view the tree graph:
#     - Right-click Result List.
#     - Click Visualize Tree.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\cheth\OneDrive\Documents\6th sem\GermanCredit.csv")

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Features and Target
X = data.drop("status", axis=1)
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Model:")
print(model)

print("\nAccuracy:", accuracy)
print("Correctly Classified:", accuracy * 100, "%")

plt.figure(figsize=(15,10))
plot_tree(
    model,
    feature_names=X.columns,
   
    filled=True
)
plt.show()

# Viva Questions

# 1. What is a Decision Tree?
# Answer: A Decision Tree is a supervised machine learning algorithm used for classification and prediction.

# 2. Why did you use Label Encoding?
# Answer: To convert categorical (text) values into numbers.

# 3. What is train_test_split()?
# Answer: It divides the dataset into training and testing datasets.

# 4. Why did you use 80% training and 20% testing?
# Answer: To train the model on most of the data while reserving some data for testing.

# 5. What does model.fit() do?
# Answer: It trains the Decision Tree model.

# 6. What does model.predict() do?
# Answer: It predicts the class labels for the test data.

# 7. What is accuracy_score()?
# Answer: It calculates the percentage of correctly classified instances.

# 8. What was your accuracy?
# Answer: Example: 67%.

# 9. What is the root node?
# Answer: The first node of the Decision Tree where the initial split is made.

# 10. What is a leaf node?
# Answer: The final node that gives the predicted class.

# 11. What does plot_tree() do?
# Answer: It displays the Decision Tree graphically.

# 12. What are features?
# Answer: The input attributes used for prediction.

# 13. What is the target variable?
# Answer: The output column (credit_risk or class) that the model predicts.

# 14. Which algorithm did you use?
# Answer: Decision Tree Classifier.

# 15. Why is the Decision Tree popular?
# Answer: Because it is simple, easy to interpret, and works well for classification problems.
