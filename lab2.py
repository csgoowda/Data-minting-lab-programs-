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
