from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model and TF-IDF vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return jsonify({
        "message": "Email Spam Detection Backend is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    email = data.get("email", "")

    if not email:
        return jsonify({
            "error": "Email message is required"
        }), 400

    # Convert email text into TF-IDF features
    email_tfidf = vectorizer.transform([email])

    # Predict
    prediction = model.predict(email_tfidf)[0]

    # Your model uses 1 = spam, 0 = normal
    if prediction == 1:
        result = "SPAM EMAIL"
    else:
        result = "NORMAL EMAIL"

    return jsonify({
        "email": email,
        "result": result
    })


if __name__ == "__main__":
    app.run(debug=True)