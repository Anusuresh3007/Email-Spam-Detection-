import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/spam_emails.csv", encoding="latin-1")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Display column names
print("\nColumn names:")
print(data.columns.tolist())

# Display dataset information
print("\nDataset information:")
print(data.info())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Display label counts
print("\nLabel distribution:")
print(data["label"].value_counts())