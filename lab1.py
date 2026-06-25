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
