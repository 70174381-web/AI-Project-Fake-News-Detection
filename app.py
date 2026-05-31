from flask import Flask, render_template, request
import joblib
import re
import string

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing — must match Fidia's pipeline exactly
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)         # remove URLs
    text = re.sub(r"\d+", "", text)                     # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    return text

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    # Preprocess before vectorizing
    clean = preprocess(news)

    # Convert text to numbers
    data = vectorizer.transform([clean])

    # Prediction
    prediction = model.predict(data)[0]

    # Result logic
    if prediction == 'FAKE':
        result = "🔴 Fake News Detected"
    else:
        result = "🟢 Real News Detected"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
    