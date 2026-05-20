import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model_lr.pkl")

# Strict Loading Mode - Loading your newly generated clean files
try:
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("✅ System Status: Model and Vectorizer loaded successfully!")
    is_model_ready = True
except Exception as e:
    print(f"❌ System Status: Error loading model files. Details: {e}")
    vectorizer = None
    model = None
    is_model_ready = False


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form.get("news", "").strip()

    if not news:
        return render_template(
            "index.html",
            prediction="⚠️ Please enter some text to analyze.",
            result_class="warning",
        )

    if not is_model_ready:
        return render_template(
            "index.html",
            prediction="❌ System Error: Machine Learning model files are missing or corrupted.",
            result_class="danger",
            old_text=news,
        )

    try:
        data = vectorizer.transform([news])
        prediction = model.predict(data)[0]

        confidence_text = ""
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(data)[0]
            confidence = probabilities[prediction] * 100
            confidence_text = f" ({confidence:.1f}% Confidence)"

        if prediction == 1:
            result = f"Fake News ❌{confidence_text}"
            result_class = "danger"
        else:
            result = f"Real News ✅{confidence_text}"
            result_class = "success"

    except Exception as e:
        result = f"❌ Prediction Error: {e}"
        result_class = "danger"

    return render_template(
        "index.html", prediction=result, result_class=result_class, old_text=news
    )


if __name__ == "__main__":
    app.run(debug=True)