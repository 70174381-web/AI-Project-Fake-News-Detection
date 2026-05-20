import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model + vectorizer safely
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError:
    print(
        "Error: 'model.pkl' or 'vectorizer.pkl' not found. Please check your file paths."
    )
    model = None
    vectorizer = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return render_template(
            "index.html",
            prediction="Model configuration error.",
            result_class="danger",
        )

    news = request.form.get("news", "").strip()

    # 1. Input Validation
    if not news:
        return render_template(
            "index.html",
            prediction="⚠️ Please enter some text to analyze.",
            result_class="warning",
        )

    # Convert text to numbers
    data = vectorizer.transform([news])

    # Prediction
    prediction = model.predict(data)[0]

    # 2. Confidence Score Logic (if supported by your model)
    confidence_text = ""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[0]
        confidence = probabilities[prediction] * 100
        confidence_text = f" ({confidence:.1f}% Confidence)"

    # 3. Dynamic Result & CSS Class Logic
    if prediction == 1:
        result = f"Fake News ❌{confidence_text}"
        result_class = "danger"
    else:
        result = f"Real News ✅{confidence_text}"
        result_class = "success"

    return render_template(
        "index.html", prediction=result, result_class=result_class, old_text=news
    )


if __name__ == "__main__":
    # debug mode controlled via environment variable safely
    is_debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=is_debug)