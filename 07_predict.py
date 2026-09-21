import joblib
import re

# Load saved model
model = joblib.load("model/spam_model.pkl")

# Load saved TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Function to clean email
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Test email
email = input("Enter an email message: ")

# Clean the email
cleaned_email = clean_text(email)

# Convert text into TF-IDF
email_tfidf = vectorizer.transform([cleaned_email])

# Predict
prediction = model.predict(email_tfidf)[0]

# Display result
if prediction == 1:
    print("\nResult: SPAM EMAIL")
else:
    print("\nResult: NORMAL EMAIL")