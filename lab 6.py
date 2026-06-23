
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r"C:\Users\cheth\Downloads\GermanCredit.csv")

# Select required attributes
data = data.iloc[:, [1, 2, 4, 6, 9, 15, 20]]

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Features and Target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Calculate Accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)
print("Correctly Classified:", accuracy * 100, "%")
print("Wrongly Classified:", (1 - accuracy) * 100, "%")
