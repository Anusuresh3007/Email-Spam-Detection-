import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load dataset
data = pd.read_csv(
    "dataset/spam_emails.csv",
    encoding="latin-1"
)

# Remove missing values
data = data.dropna(subset=["email", "label"])


# Clean email text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data["email"] = data["email"].apply(clean_text)


# Input and output
X = data["email"]
y = data["label"]


# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TF-IDF
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)


# Train model
model = LogisticRegression()

model.fit(X_train_tfidf, y_train)


# Save model and vectorizer
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("Model saved successfully!")
print("Saved: model/spam_model.pkl")
print("Saved: model/tfidf_vectorizer.pkl")