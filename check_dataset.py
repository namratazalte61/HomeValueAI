import pandas as pd

# Load dataset
data = pd.read_csv("dataset/House_Price_Data.csv")

# Show first 5 rows
print(data.head())

# Show column names
print("\nColumns:")
print(data.columns)

# Show dataset information
print("\nDataset Information:")
print(data.info())