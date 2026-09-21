import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset/spam_emails.csv", encoding="latin-1")

# Remove missing values
data = data.dropna(subset=["email", "label"])

# Clean email text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["email"] = data["email"].apply(clean_text)

# Separate input and output
X = data["email"]
y = data["label"]

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display results
print("Total emails:", len(data))
print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())