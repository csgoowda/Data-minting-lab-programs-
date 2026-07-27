# Question 6: Train Decision Tree using selected attributes and compare performance.
# Aim: Check if fewer selected attributes can still provide good results.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("GermanCredit.csv")

# Select Required Attributes
data = data.iloc[:, [1, 2, 4, 6, 9, 15, 20]]

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Features and Target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

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
print("Wrongly Classified:", (1 - accuracy) * 100, "%")



# What is the Question Asking?
# The question asks whether using only a few selected attributes is
# enough to build a good Decision Tree.
# You compare the accuracy with the model that used all attributes.

# Conclusion
# Using fewer important attributes makes the model simpler and faster,
# but the accuracy may decrease slightly because some useful information
# is removed.

# Viva Questions

# 1. What is Feature Selection?
# Answer: Selecting only the important attributes for building the model.

# 2. Why did you use only selected attributes?
# Answer: To check whether fewer attributes can still produce good accuracy.

# 3. Why did the accuracy decrease?
# Answer: Because some useful information was removed when fewer attributes were used.

# 4. What are the advantages of Feature Selection?
# Answer:
# - Faster training
# - Simpler model
# - Lower memory usage
# - Easier interpretation

# 5. Can Feature Selection improve accuracy?
# Answer: Yes, if it removes irrelevant or noisy attributes.

# 6. Which classifier did you use?
# Answer: Decision Tree Classifier.

# 7. What is the class (target) attribute?
# Answer: credit_risk (or class, depending on the dataset).

# 8. Why compare this model with the previous one?
# Answer: To see how much accuracy changes when using fewer attributes.

# 9. What is the conclusion?
# Answer: A model with fewer important attributes is simpler and faster,
# but its accuracy may be slightly lower than using all attributes.

# 10. Is a simpler model always better?
# Answer: Not always. A simpler model is easier to understand, but it may
# lose some predictive accuracy if important attributes are removed.
