import pandas as pd
import re

# Load the dataset
data = pd.read_csv("dataset/spam_emails.csv", encoding="latin-1")

# Remove missing values
data = data.dropna(subset=["email", "label"])

# Convert email text to lowercase
data["email"] = data["email"].str.lower()

# Clean the email text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["email"] = data["email"].apply(clean_text)

# Display cleaned data
print("Cleaned emails:")
print(data.head())

# Display dataset size
print("\nDataset shape:")
print(data.shape)

# Display label distribution
print("\nLabel distribution:")
print(data["label"].value_counts())