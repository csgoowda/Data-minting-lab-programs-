# Question 2: What attributes are crucial in making credit assessment?
# Aim: Use a Decision Tree's feature importance to identify important attributes.

# WEKA Steps
# 1. Open WEKA Explorer.
# 2. Load credit-g.arff.
# 3. Go to the Select attributes tab.
# 4. Choose Attribute Evaluator → InfoGainAttributeEval.
# 5. Choose Search Method → Ranker.
# 6. Click Start.
# 7. WEKA displays the ranked attributes.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load Dataset
dataset = pd.read_csv('credit-g_csv.csv')

# Label Encoding
label_encoder = LabelEncoder()

for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column] = label_encoder.fit_transform(dataset[column])

# Features and Target
X = dataset.drop('class', axis=1)
y = dataset['class']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=6)
model.fit(X_train, y_train)

# Feature Importance
importance = model.feature_importances_

# Store IG Values
attribute_ig = {}

for i, feature in enumerate(X.columns):
    attribute_ig[feature] = importance[i]

# Sort by Information Gain
sorted_ig = sorted(
    attribute_ig.items(),
    key=lambda x: x[1],
    reverse=True
)

print("Top Important Attributes:\n")

for attribute, ig in sorted_ig[:6]:
    print(attribute, ":", ig)


this is correct


for attribute, ig in sorted_ig[:6]:
    print(attribute, ":", ig)

# Viva Questions & Answers

# 1. What is Information Gain?
# Answer: Information Gain measures how useful an attribute is for classification.

# 2. Why do we use Information Gain?
# Answer: To identify the most important attributes for prediction.

# 3. Which attribute had the highest Information Gain?
# Answer: checking_status

# 4. What does a higher Information Gain mean?
# Answer: It means the attribute is more important for prediction.

# 5. Which function gives the importance of attributes?
# Answer: model.feature_importances_

# 6. Why did you sort the attributes?
# Answer: To display the attributes from the most important to the least important.

# 7. Why did you print only the top six attributes?
# Answer: Because the question asks us to identify the most important attributes,
# and the top six have the highest importance values.

# 8. What is Feature Importance?
# Answer: It indicates how much each attribute contributes to the Decision Tree's predictions.

# 9. Which algorithm did you use?
# Answer: Decision Tree Classifier.

# 10. What is the conclusion of this experiment?
# Answer: The Decision Tree identified the most important attributes for credit assessment,
# with checking_status being the most influential. These important attributes can be used
# to build simpler and effective prediction models.
