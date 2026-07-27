# Question 1: List all the categorical (nominal) attributes and the real-valued attributes separately.
# Aim: Read the dataset and print categorical vs numerical columns.

import pandas as pd

data = pd.read_csv("credit-g_csv.csv")

print("Categorical Attributes:\n")

for col in data.columns:
    if data[col].dtype == 'object':
        print(col)

print("\nReal-Valued Attributes:\n")

for col in data.columns:
    if data[col].dtype != 'object':
        print(col)


# WEKA Steps
# 1. Open WEKA Explorer.
# 2. Click Open File.
# 3. Load credit-g.arff.
# 4. Go to the Preprocess tab.
# 5. WEKA lists all attributes.
# 6. Attributes with Type = Nominal are categorical.
# 7. Attributes with Type = Numeric are real-valued.
